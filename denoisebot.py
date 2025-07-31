import torch
import torch.nn as nn
from discord import Intents, Embed, File, Client, Message
from discord.ext import commands
from PIL import Image
import io
from torchvision import transforms
import matplotlib.pyplot as plt

# ---Bot Setup--- #

intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)


class Autoencoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1), # 32x32x3 -> 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1), # 16x16x16 -> 16x16x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 16x16x32 -> 16x16x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0), # 16x16x64 -> 16x16x128
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1, padding=0), # 16x16x128 -> 16x16x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), # 16x16x64 -> 16x16x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=1), # 16x16x32 -> 16x16x16
            nn.ReLU(), 
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1), # 16x16x16 -> 32x32x3
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x) 
        return x
    
# ---Model setup--- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
model.eval()

def process_image(image_bytes, num_iterations):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    original_image = transform(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    image = original_image.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(num_iterations):
            image = model(image)

    reconstructed_image = transforms.ToPILImage()(image.squeeze().cpu())
    return reconstructed_image

async def send_message(message: Message, user_message: str):
    if not user_message:
        return
    
    try:
        response = get_response(user_message)
        await message.channel.send(response)
    except Exception as e:
        await message.channel.send(f"An error occurred: {e}")


def get_response(user_message: str):
    lowered = user_message.lower()
    if lowered.__contains__("help"):
        return "To use this bot, type insert an image and type how many denoising iterations you would like to run."
    else:
        return "Please type \"help\" for more information on how to use this bot."
    
@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message: Message):
    if message.author == client.user:
        return

    username = message.author
    user_message = message.content
    channel = message.channel
    print(f"Received message from {username}: {user_message}")
    
    # CLEAR COMMAND
    if user_message == "!clear" or user_message == "!CLEAR":
        if message.author.guild_permissions.manage_messages:
            await message.channel.purge()
            await message.channel.send("Cleared all messages.", delete_after=5)
        else:
            await message.channel.send("You do not have permission to clear messages.", delete_after=5)

    if message.attachments:
        try:
            num_iterations = int(user_message.split()[-1])
            if num_iterations < 1:
                await channel.send("The number of iterations must be at least 1.")
                return
        except ValueError:
            await channel.send("Please provide a valid number of iterations.")
            return

        attachment = message.attachments[0]
        if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_bytes = await attachment.read()
            reconstructed_image = process_image(image_bytes, num_iterations)

            image_buffer = io.BytesIO()
            reconstructed_image.save(image_buffer, format='PNG')
            image_buffer.seek(0)

            # Send the denoised image back to the user
            await channel.send(file=File(fp=image_buffer, filename='reconstructed_image.png'))
        else:
            await channel.send("Please upload a valid image file (png, jpg, jpeg).")
    else:
        await send_message(message, user_message)

def main():
    client.run(token=TOKEN)

if __name__ == "__main__":
    main()