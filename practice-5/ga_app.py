from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import ga_core
import db_utils


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_utils.setup_database('ga.db')
    yield


app = FastAPI(
    title="Генетичний алгоритм",
    description="Simple Genetic Algorithm для оптимізації функції",
    lifespan=lifespan
)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run")
async def run_ga(
    request: Request,
    pop_size: int = Form(...),
    num_generations: int = Form(...),
    crossover_prob: float = Form(...),
    mutation_prob: float = Form(...),
    tournament_size: int = Form(...)
):
    params = {
        'pop_size': pop_size,
        'num_generations': num_generations,
        'crossover_prob': crossover_prob,
        'mutation_prob': mutation_prob,
        'tournament_size': tournament_size
    }

    best_individual, best_fitness = ga_core.run_genetic_algorithm(params)

    return RedirectResponse(url="/results", status_code=303)


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    try:
        runs = db_utils.get_all_runs('ga.db')
    except Exception:
        runs = []
    return templates.TemplateResponse(
        "results.html", {"request": request, "runs": runs}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ga_app:app", host="0.0.0.0", port=5000, reload=True)
