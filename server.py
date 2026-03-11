import pandas as pd
import numpy as np
import random
import uuid
import io
import base64
import os
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="GeneticFS Engine Pro")

# --- Global State ---
jobs = {}

# --- GA Setup ---
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def get_advanced_scores(X_train, y_train, task_type):
    """Combines Statistical Spearman and Model-based RF Importance."""
    # 1. Spearman (captures non-linear rank relationships)
    corr_scores = X_train.corrwith(y_train, method='spearman').abs()
    corr_scores = np.nan_to_num(corr_scores / (corr_scores.max() if corr_scores.max() != 0 else 1))
    
    # 2. Random Forest (captures complex interactions)
    if task_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    rf_scores = pd.Series(model.feature_importances_, index=X_train.columns)
    rf_scores = rf_scores / (rf_scores.max() if rf_scores.max() != 0 else 1)
    
    # Combined score (weighted toward RF for smarter selection)
    return (0.4 * corr_scores) + (0.6 * rf_scores)

def evaluate(individual, X_train, y_train, feature_names, importance_scores):
    """Ruthless evaluation: Heavy penalties for noise and redundancy."""
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # 1. Heavy penalty for selecting nothing
    if len(selected_features) == 0:
        return -50.0,

    # 2. Relevance Score (The 'Reward')
    relevance = np.sum(importance_scores[selected_features])

    # 3. Redundancy Penalty (The 'Filter')
    # Squaring the mean correlation between features to punish overlap heavily
    if len(selected_features) > 1:
        X_selected = X_train[selected_features]
        corr_matrix = X_selected.corr().abs().values
        redundancy_vals = corr_matrix[np.triu_indices(len(selected_features), k=1)]
        avg_redundancy = np.mean(np.nan_to_num(redundancy_vals))
        redundancy_penalty = (avg_redundancy ** 2) * 8.0 
    else:
        redundancy_penalty = 0

    # 4. Sparsity Pressure (The 'Dumper')
    # Increased to 0.15: A feature must provide significant value to be worth its cost
    sparsity_penalty = 0.15 * len(selected_features)

    fitness = relevance - redundancy_penalty - sparsity_penalty
    return float(np.nan_to_num(fitness)),

def target_encode(X, y, column):
    X[column] = X[column].fillna('Missing')
    temp_df = pd.concat([X, y], axis=1)
    mean_encoding = temp_df.groupby(column)[y.name].mean()
    X[column] = X[column].map(mean_encoding)
    return X

# --- Background Task ---
def run_evolution_task(job_id, df, target_column, task_type, n_gen, pop_size, cxpb, mutpb):
    try:
        # Data Cleaning
        df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # --- PRE-FLIGHT FILTER: REMOVE CONSTANT JUNK ---
        X = X.loc[:, X.nunique() > 1]
        
        # Handle Target (String to Numeric)
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str).str.strip()), index=y.index, name=y.name)
        
        # Handle Features (Categorical)
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            X = target_encode(X, y, col)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        feature_names = X_train.columns.tolist()
        importance_scores = get_advanced_scores(X_train, y_train, task_type)

        # DEAP Setup
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(feature_names))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate, X_train=X_train, y_train=y_train, 
                         feature_names=feature_names, importance_scores=importance_scores)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
        toolbox.register("select", tools.selTournament, tournsize=7)

        pop = toolbox.population(n=pop_size)
        
        for gen in range(n_gen):
            offspring = tools.selTournament(pop, len(pop), tournsize=7)
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            best_ind = tools.selBest(pop, 1)[0]
            
            # JSON-safe logging
            fit_val = float(best_ind.fitness.values[0])
            jobs[job_id]["generation_log"].append({
                "gen": gen,
                "best_fitness": round(fit_val if np.isfinite(fit_val) else 0.0, 4),
                "avg_features": round(float(np.mean([sum(ind) for ind in pop])), 1)
            })
            jobs[job_id]["progress"] = int(((gen + 1) / n_gen) * 100)

        best_individual = tools.selBest(pop, 1)[0]
        selected_features = [feature_names[i] for i in range(len(best_individual)) if best_individual[i] == 1]
        
        jobs[job_id].update({
            "status": "done",
            "selected_features": selected_features,
            "n_selected": len(selected_features),
            "n_total": len(feature_names),
            "score_value": round(float(best_individual.fitness.values[0]), 4),
            "score_label": "Engine Fitness Score"
        })
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

# --- FastAPI Routes ---
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return {"rows": len(df), "columns": df.columns.tolist(), "file_content": base64.b64encode(contents).decode('utf-8')}

@app.post("/run")
async def run_ga(
    background_tasks: BackgroundTasks,
    file_content: str = Form(...),
    target_column: str = Form(...),
    task_type: str = Form(...),
    n_gen: int = Form(...),
    pop_size: int = Form(...),
    cxpb: float = Form(...),
    mutpb: float = Form(...)
):
    job_id = str(uuid.uuid4())
    df = pd.read_csv(io.BytesIO(base64.b64decode(file_content)))
    jobs[job_id] = {"status": "running", "progress": 0, "generation_log": []}
    background_tasks.add_task(run_evolution_task, job_id, df, target_column, task_type, n_gen, pop_size, cxpb, mutpb)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    return jobs.get(job_id, {"error": "Job not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)