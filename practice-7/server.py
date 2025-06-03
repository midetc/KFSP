from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from model import EpidemicModel
from templates import get_html_template
from config import SERVER_CONFIG

app = FastAPI(title="Клітинний автомат: Модель епідемії")
model = EpidemicModel()


@app.get("/")
async def get_index():
    return HTMLResponse(content=get_html_template())


@app.post("/reset")
async def reset_model(params: dict):
    global model
    size = params.get('size', 50)
    p_infect = params.get('p_infect', 0.3)
    t_recover = params.get('t_recover', 10)
    model = EpidemicModel(size, p_infect, t_recover)
    
    image_base64 = model.to_image_base64()
    stats = model.get_stats()
    
    return JSONResponse({
        'image': image_base64,
        'stats': stats
    })


@app.post("/step")
async def step_model():
    model.step()
    image_base64 = model.to_image_base64()
    stats = model.get_stats()
    
    return JSONResponse({
        'image': image_base64,
        'stats': stats
    })


@app.get("/current")
async def get_current():
    image_base64 = model.to_image_base64()
    stats = model.get_stats()
    
    return JSONResponse({
        'image': image_base64,
        'stats': stats
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_CONFIG['host'],
                port=SERVER_CONFIG['port']) 