from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def create_product(product: dict):
    return {"message": f"Product {product['name']} created!"}

@router.get("/")
def get_products():
    return {"products": ["Product 1", "Product 2"]}
