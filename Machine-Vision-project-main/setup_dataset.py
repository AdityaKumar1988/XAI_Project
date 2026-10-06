# Rajat Srivastava - 23BAI0125
import os
from PIL import Image, ImageDraw, ImageFont
import random

# Create folders
base_path = "dataset"
folders = [
    "genuine", "fake"
]

for f in folders:
    os.makedirs(os.path.join(base_path, f), exist_ok=True)

# Try loading font
try:
    # Use standard system fonts for Mac if possible, or fall back
    font_paths = ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "arial.ttf"]
    font = None
    for p in font_paths:
        if os.path.exists(p):
            font = ImageFont.truetype(p, 28)
            bold_font = ImageFont.truetype(p, 32)
            break
    if font is None:
        font = ImageFont.load_default()
        bold_font = font
except:
    font = ImageFont.load_default()
    bold_font = font

names = ["Rajat", "Aman", "Priya", "Sneha", "Rahul", "Karan", "Neha", "Arjun", "Simran", "Vikas"]

def create_authentic(name, cgpa):
    img = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(img)

    y = 50
    draw.text((150, y), "ACADEMIC RESULT SHEET", font=bold_font, fill="black")
    y += 80

    draw.text((50, y), f"Name: {name}", font=font, fill="black")
    y += 50
    draw.text((50, y), f"Reg No: 23BAI{random.randint(1000,9999)}", font=font, fill="black")
    y += 50

    draw.text((50, y), "Course: B.Tech AIML", font=font, fill="black")
    y += 100

    subjects = ["DSA", "DBMS", "ML", "OS", "Maths"]

    for sub in subjects:
        grade = random.choice(["A", "B+", "A+", "B"])
        draw.text((50, y), f"{sub} : {grade}", font=font, fill="black")
        y += 50

    y += 50
    draw.text((50, y), f"CGPA: {cgpa}", font=font, fill="black")

    return img


def create_fake_versions(img, original_cgpa):
    fake_images = []

    # 1. CGPA tampered
    img1 = img.copy()
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([50, 650, 400, 700], fill="white") # Clear old CGPA
    draw1.text((50, 650), f"CGPA: {round(random.uniform(9.5, 9.9),2)}", font=font, fill="black")
    fake_images.append(img1)

    # 2. Font mismatch
    img2 = img.copy()
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([50, 650, 400, 700], fill="white") # Clear old CGPA
    draw2.text((50, 650), f"CGPA: {original_cgpa}", font=bold_font, fill="black")
    fake_images.append(img2)

    # 3. Alignment shift
    img3 = img.copy()
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([50, 650, 400, 700], fill="white") # Clear old CGPA
    draw3.text((200, 650), f"CGPA: {original_cgpa}", font=font, fill="black")
    fake_images.append(img3)

    return fake_images


# Generate dataset
for i in range(10):
    name = names[i]
    cgpa = round(random.uniform(6.5, 8.8), 2)

    authentic_img = create_authentic(name, cgpa)

    auth_path = f"{base_path}/genuine/doc_{i}.png"
    fake_dir = f"{base_path}/fake/"

    authentic_img.save(auth_path)

    fake_imgs = create_fake_versions(authentic_img, cgpa)

    for j, f_img in enumerate(fake_imgs):
        f_img.save(f"{fake_dir}/doc_{i}_fake{j}.png")

print("✅ Dataset Generated Successfully!")