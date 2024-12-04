import requests

url = "http://127.0.0.1:8000/generate-content"

data = {
    "text": """Calvin Klein is one of the world’s leading global fashion lifestyle brands with a history of bold, non-conformist ideals that inform everything we do. Founded in New York in 1968, the brand’s minimalist and sensual aesthetic drives our approach to product design and communication, creating a canvas that offers the possibility of limitless self-expression. The Calvin Klein brands – CK Calvin Klein, Calvin Klein, Calvin Klein Jeans, Calvin Klein Underwear, and Calvin Klein Performance – are connected by the intention and purpose of elevating everyday essentials to globally iconic status. Each of the brands has a distinct identity and position in the retail landscape, providing us the opportunity to market a range of universally appealing products to domestic and international consumers with a variety of needs. Our products are underpinned by responsible design, high-quality construction, and the elimination of all unnecessary details. We strive for unique and dimensional pieces that continuously wear well and remain relevant season after season. Global retail sales of Calvin Klein products were approximately $8.5 billion in 2021.

Calvin Klein continues to solidify its position as an innovator of emerging digital platforms and modern marketing campaigns. PVH acquired Calvin Klein in 2003 and continues to oversee a focused approach to growing the brand’s worldwide relevance, presence, and long term growth.
Generate 4 posts for Instagram and only 2 posts for Twitter for Winter Apparels""",
    "platforms": ["Instagram","Twitter"]
}

response = requests.post(url, json=data)