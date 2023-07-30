from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from io import BytesIO
import logging
import pydantic
import typing as t
from functools import lru_cache

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class ImageQuery(pydantic.BaseModel):
    question: str

class ImageQueryResponse(pydantic.BaseModel):
    question: str
    answer: str

@lru_cache
def load_model():
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    return model, processor

def process_queries(image, query):
    model, processor = load_model()
    encoding = processor(image, query, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    elements=model.config.id2label[idx]
    
    return elements

def convert_png_to_jpg(image):
    rgb_image = image.convert('RGB')
    byte_arr = BytesIO()
    rgb_image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return Image.open(byte_arr)


def query_image(image_file: str, queries: t.List[ImageQuery]) -> t.List[ImageQueryResponse]:
    image = Image.open(image_file)
    
    # ViLT model only supports JPG images
    if image.format == 'PNG':
        image = convert_png_to_jpg(image)

    responses = []

    for query in queries:
        answer = process_queries(image, query.question)
        responses.append(
            ImageQueryResponse(
                question=query.question,
                answer=answer
            )
        )
    
    return responses
    
        
def display_results(title, responses):
    
    print(title+"\n")
    
    for response in responses:
        print(f"**{response.question}**")
        print(response.answer)
        
    print("\n")


if __name__ == "__main__":
    
    queries = [
        ImageQuery(question="What is the primary subject of the painting?"),
        ImageQuery(question="What are the seocndary subjects of the painting?"),
        ImageQuery(question="What is the style of the painting?"),
        ImageQuery(question="What is the medium of the painting?"),
        ImageQuery(question="What is the genre of the painting?"),
        ImageQuery(question="What is the time period of the painting?"),
        ImageQuery(question="What is the artist's name?"),
    ]
    
    display_results("Monet ", query_image("monet.jpg", queries))
    display_results("Picasso ", query_image("picasso.jpg", queries))
    display_results("Pointillism ",query_image("pointillism.jpg", queries))
    display_results("Van Gogh", query_image("vangough.webp", queries))
    
    