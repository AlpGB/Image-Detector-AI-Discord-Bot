import discord
from discord.ext import commands
import os
import requests
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image, ImageDraw

intents = discord.Intents.all()
intents.messages = True
intents.message_content = True
prefix = '!'
bot = commands.Bot(command_prefix=prefix, intents=intents)

DEEPAI_API_KEY = 'your DeepAI API Key'
IMAGE_GENERATION_API_URL = 'https://api.deepai.org/api/text2img'
IMAGE_DIR = "/Users/evrimbayrak/Desktop/AI Bot Folder/"

model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

transform = T.Compose([
    T.ToTensor()
])

@bot.event
async def on_ready():
    print(f'{bot.user} is now online')

def detect_and_draw_objects(image_path, output_image_path):
    image = Image.open(image_path)

    if image.mode == "L":  
        image = image.convert("RGB")
    elif image.mode == "RGBA":  
        image = image.convert("RGB")
    
    image_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        predictions = model(image_tensor)

    draw = ImageDraw.Draw(image)

    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > 0.5:  
            box = box.tolist()
            draw.rectangle(box, outline="green", width=3)

    image.save(output_image_path)

@bot.command(name='detect')
async def detect_objects_command(ctx):
    if ctx.message.attachments:
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]

            if attachment.filename.endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(IMAGE_DIR, attachment.filename)
                output_image_path = os.path.join(IMAGE_DIR, f"detected_{attachment.filename}")

                await attachment.save(image_path)

                try:
                    detect_and_draw_objects(image_path, output_image_path)

                    await ctx.send("Objects detected!", file=discord.File(output_image_path))
                except Exception as e:
                    await ctx.send(f"An error occurred: {e}")
                finally:
                    if os.path.isfile(image_path):
                        os.remove(image_path)
                    if os.path.isfile(output_image_path):
                        os.remove(output_image_path)
            else:
                await ctx.send("Please attach a valid image file (PNG/JPG).")
    else:
        await ctx.send("Please attach an image for object detection.")

@bot.command(name='foto')
async def generate_image(ctx, *, description: str):
    try:
        print(f"Using API Key: {DEEPAI_API_KEY}")

        r = requests.post(
            "https://api.deepai.org/api/text2img",
            data={
                'text': description,
            },
            headers={'Api-Key': DEEPAI_API_KEY})

        print(r.status_code)
        print(r.text)

        if r.status_code == 200:
            data = r.json()
            image_url = data.get('output_url')

            if image_url:
                await ctx.send(f"Here is your generated image: {image_url}")
            else:
                await ctx.send("Failed to generate the image.")
        else:
            await ctx.send(f"API returned an error: {r.status_code} - {r.text}")
    
    except requests.exceptions.RequestException as e:
        await ctx.send(f"API request failed: {e}")
    except Exception as e:
        await ctx.send(f"An unexpected error occurred: {e}")

bot.run('your bot token')
