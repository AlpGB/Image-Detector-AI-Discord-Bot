import discord
from discord.ext import commands
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import os

intents = discord.Intents.all()
intents.messages = True
intents.message_content = True

prefix = '!'
bot = commands.Bot(command_prefix=prefix, intents=intents)

# Directory for saving images
IMAGE_DIR = "/Users/evrimbayrak/Desktop/AI Bot Folder/"

# Load the Faster R-CNN model with pretrained weights
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define a set of transformations to apply to the image
transform = T.Compose([
    T.ToTensor()
])

# Event when the bot is ready
@bot.event
async def on_ready():
    print(f'{bot.user} is now online')

# Function to detect objects in an image using Faster R-CNN and draw bounding boxes
def detect_and_draw_objects(image_path, output_image_path):
    # Open the image
    image = Image.open(image_path)

    # Automatically convert the image to RGB if necessary
    if image.mode == "L":  # Grayscale (1 channel)
        image = image.convert("RGB")
    elif image.mode == "RGBA":  # RGBA (4 channels)
        image = image.convert("RGB")
    
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        predictions = model(image_tensor)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > 0.5:  # Draw boxes only for confident predictions
            # Box coordinates
            box = box.tolist()
            # Draw a green rectangle (bounding box)
            draw.rectangle(box, outline="green", width=3)

    # Save the image with bounding boxes
    image.save(output_image_path)

# Command to detect objects in an attached image and draw bounding boxes
@bot.command(name='detect')
async def detect_objects_command(ctx):
    # Check if the message has attachments
    if ctx.message.attachments:
        # Ensure the message has at least one attachment
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            
            # Ensure the attachment is an image
            if attachment.filename.endswith(('png', 'jpg', 'jpeg')):
                # Path to save the image
                image_path = os.path.join(IMAGE_DIR, attachment.filename)
                output_image_path = os.path.join(IMAGE_DIR, f"detected_{attachment.filename}")

                # Save the image locally
                await attachment.save(image_path)

                try:
                    # Detect objects and draw bounding boxes in the image
                    detect_and_draw_objects(image_path, output_image_path)

                    # Send the processed image with detected objects back to the channel
                    await ctx.send("Objects detected!", file=discord.File(output_image_path))
                except Exception as e:
                    await ctx.send(f"An error occurred: {e}")
                finally:
                    # Clean up by removing the saved images
                    if os.path.isfile(image_path):
                        os.remove(image_path)
                    if os.path.isfile(output_image_path):
                        os.remove(output_image_path)
            else:
                await ctx.send("Please attach a valid image file (PNG/JPG).")
        else:
            await ctx.send("Please attach an image for object detection.")
    else:
        await ctx.send("Please attach an image for object detection.")

bot.run('bot token')

