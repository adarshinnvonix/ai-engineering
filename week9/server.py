"""
A simple MCP server built with FastMCP. It exposes a small product catalog:
  - a RESOURCE: the full catalog, readable as one block of data
  - two TOOLS: search by keyword, and get full details for one product by ID
"""

from fastmcp import FastMCP

mcp = FastMCP("ProductCatalog")

PRODUCTS = [
    {"id": 1, "name": "Wireless Mouse", "category": "Electronics", "price": 25.99, "stock": 120},
    {"id": 2, "name": "Mechanical Keyboard", "category": "Electronics", "price": 89.99, "stock": 45},
    {"id": 3, "name": "Office Chair", "category": "Furniture", "price": 149.99, "stock": 30},
    {"id": 4, "name": "Standing Desk", "category": "Furniture", "price": 299.99, "stock": 15},
    {"id": 5, "name": "USB-C Hub", "category": "Electronics", "price": 39.99, "stock": 80},
    {"id": 6, "name": "Desk Lamp", "category": "Furniture", "price": 19.99, "stock": 200},
]


# A RESOURCE - data a client can read directly, like opening a file.
@mcp.resource("catalog://products")
def get_full_catalog() -> str:
    """The entire product catalog, one line per product."""
    return "\n".join(
        f"#{p['id']} {p['name']} - {p['category']} - ${p['price']} ({p['stock']} in stock)"
        for p in PRODUCTS
    )


# TOOLS - actions a client (or an LLM through an agent) can call with arguments.
@mcp.tool()
def search_products(query: str) -> str:
    """Search products by name or category keyword (e.g. 'desk', 'electronics')."""
    query = query.lower()
    matches = [p for p in PRODUCTS if query in p["name"].lower() or query in p["category"].lower()]
    if not matches:
        return "No matching products found."
    return "\n".join(f"#{p['id']} {p['name']} - ${p['price']} ({p['stock']} in stock)" for p in matches)


@mcp.tool()
def get_product_details(product_id: int) -> str:
    """Get full details for one product, given its ID."""
    for p in PRODUCTS:
        if p["id"] == product_id:
            return str(p)
    return f"No product found with id {product_id}."


if __name__ == "__main__":
    mcp.run()  # starts the server, listening over stdio by default