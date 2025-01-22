from fastapi import FastAPI
from app.routes import users, products

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "나이스캐치에 오신 것을 환영합니다."}

# 라우터 등록
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(products.router, prefix="/products", tags=["Products"])