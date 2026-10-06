"""
Academic Document Forgery Benchmark Dataset Generator & Splitter
Generates verified academic result sheets / transcripts matching labels.csv specification.
Produces:
- 600 high-resolution document images (300 genuine, 300 forged across 7 fraud types)
- Exact pixel-level binary ground-truth masks (0 = authentic, 255 = tampered)
- Leakage-safe grouped splits (train.csv, val.csv, test.csv) by master document ID
"""

import os
import csv
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def get_fonts():
    """Load system fonts or fall back to default PIL font."""
    font_paths = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    font_regular = None
    font_bold = None
    font_title = None
    font_small = None
    
    for p in font_paths:
        if os.path.exists(p):
            try:
                font_title = ImageFont.truetype(p, 28)
                font_bold = ImageFont.truetype(p, 22)
                font_regular = ImageFont.truetype(p, 18)
                font_small = ImageFont.truetype(p, 14)
                break
            except Exception:
                continue
                
    if font_regular is None:
        font_regular = ImageFont.load_default()
        font_bold = font_regular
        font_title = font_regular
        font_small = font_regular
        
    return font_title, font_bold, font_regular, font_small

def draw_header_and_seal(draw, width=800):
    """Draws realistic university header and official seal."""
    # Header border
    draw.rectangle([40, 30, width - 40, 110], outline="#1a365d", width=2)
    # University title
    draw.text((width // 2 - 160, 45), "NATIONAL INSTITUTE OF TECHNOLOGY", fill="#1a365d")
    draw.text((width // 2 - 120, 75), "OFFICIAL ACADEMIC TRANSCRIPT", fill="#2d3748")
    
    # Official circular seal on bottom right
    cx, cy, r = width - 130, 920, 45
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#742a2a", width=3)
    draw.ellipse([cx - r + 6, cy - r + 6, cx + r - 6, cy + r - 6], outline="#742a2a", width=1)
    draw.text((cx - 28, cy - 8), "SEAL / VERIFIED", fill="#742a2a")

def render_authentic_document(meta, font_title, font_bold, font_regular, font_small, width=800, height=1050):
    """Renders a genuine academic transcript document and returns (img, metadata_coords)."""
    img = Image.new("RGB", (width, height), color="#fcfcfc")
    draw = ImageDraw.Draw(img)
    
    # Subtle background paper texture
    bg_noise = np.random.randint(-4, 5, (height, width, 3), dtype=np.int16)
    img_np = np.clip(np.array(img, dtype=np.int16) + bg_noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    
    draw_header_and_seal(draw, width)
    
    coords = {}
    
    # Student metadata block
    y = 140
    draw.rectangle([40, y, width - 40, y + 130], outline="#cbd5e1", width=1)
    
    # Student Name
    draw.text((60, y + 15), "Student Name:", fill="#475569")
    name_x, name_y = 200, y + 15
    draw.text((name_x, name_y), meta["name"], fill="#0f172a")
    coords["name"] = (name_x, name_y, 250, 26)
    
    # Registration ID
    draw.text((60, y + 45), "Register ID:", fill="#475569")
    id_x, id_y = 200, y + 45
    draw.text((id_x, id_y), meta["reg_id"], fill="#0f172a")
    coords["reg_id"] = (id_x, id_y, 200, 26)
    
    # Department / Program
    draw.text((60, y + 75), "Department:", fill="#475569")
    dept_x, dept_y = 200, y + 75
    draw.text((dept_x, dept_y), meta["department"], fill="#0f172a")
    coords["department"] = (dept_x, dept_y, 350, 26)
    
    # Academic Year
    draw.text((width - 240, y + 15), f"Batch: {meta['batch']}", fill="#475569")
    draw.text((width - 240, y + 45), f"Semester: {meta['semester']}", fill="#475569")
    
    # Subject Grades Table
    y_table = 300
    draw.rectangle([40, y_table, width - 40, y_table + 35], fill="#edf2f7")
    draw.text((60, y_table + 8), "Course Code", fill="#1a202c")
    draw.text((220, y_table + 8), "Course Title", fill="#1a202c")
    draw.text((500, y_table + 8), "Credits", fill="#1a202c")
    draw.text((640, y_table + 8), "Grade", fill="#1a202c")
    
    y_row = y_table + 45
    coords["grades"] = []
    for code, title, cr, gr in meta["courses"]:
        draw.text((60, y_row), code, fill="#334155")
        draw.text((220, y_row), title, fill="#334155")
        draw.text((515, y_row), str(cr), fill="#334155")
        draw.text((650, y_row), gr, fill="#0f172a")
        coords["grades"].append((645, y_row - 2, 45, 26, gr))
        # subtle divider line
        draw.line([40, y_row + 30, width - 40, y_row + 30], fill="#e2e8f0", width=1)
        y_row += 40
        
    # Performance summary block (CGPA & Attendance)
    y_summary = y_row + 30
    draw.rectangle([40, y_summary, width - 40, y_summary + 110], outline="#cbd5e1", width=1)
    
    draw.text((60, y_summary + 20), "Cumulative GPA (CGPA):", fill="#334155")
    gpa_x, gpa_y = 300, y_summary + 18
    draw.text((gpa_x, gpa_y), f"{meta['cgpa']:.2f} / 10.00", fill="#0f172a")
    coords["cgpa"] = (gpa_x, gpa_y - 2, 160, 30)
    
    draw.text((60, y_summary + 60), "Overall Attendance:", fill="#334155")
    att_x, att_y = 300, y_summary + 58
    draw.text((att_x, att_y), f"{meta['attendance']}%", fill="#0f172a")
    coords["attendance"] = (att_x, att_y - 2, 100, 30)
    
    # Official Signature Line
    draw.line([60, 940, 260, 940], fill="#64748b", width=1)
    draw.text((70, 950), "Controller of Examinations", fill="#64748b")
    # Date issued
    draw.text((60, 990), f"Date of Issue: 12-June-2024", fill="#94a3b8")
    
    return img, coords

def apply_forgery(base_img, coords, fraud_type, meta, font_bold, font_regular):
    """
    Applies realistic forgery to base document and generates exact ground-truth mask.
    Returns: (forged_img, mask_img)
    """
    width, height = base_img.size
    forged_img = base_img.copy()
    mask = Image.new("L", (width, height), 0)
    draw_f = ImageDraw.Draw(forged_img)
    draw_m = ImageDraw.Draw(mask)
    
    # Tamper background fill color sampled from local region
    bg_patch = "#fbfbfb"
    
    if fraud_type == "gpa_tampering":
        x, y, w, h = coords["cgpa"]
        # Clear authentic CGPA
        draw_f.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=bg_patch)
        new_cgpa = round(random.uniform(9.45, 9.95), 2)
        # Draw altered CGPA with slight font/weight mismatch
        draw_f.text((x, y), f"{new_cgpa:.2f} / 10.00", fill="#000000")
        draw_m.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=255)
        
    elif fraud_type == "attendance_edit":
        x, y, w, h = coords["attendance"]
        draw_f.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=bg_patch)
        new_att = random.randint(88, 97)
        draw_f.text((x, y), f"{new_att}%", fill="#000000")
        draw_m.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=255)
        
    elif fraud_type == "department_change":
        x, y, w, h = coords["department"]
        draw_f.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=bg_patch)
        new_dept = random.choice(["Computer Science and Engineering", "Artificial Intelligence & Data Science"])
        draw_f.text((x, y), new_dept, fill="#050505")
        draw_m.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=255)
        
    elif fraud_type == "name_change":
        x, y, w, h = coords["name"]
        draw_f.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=bg_patch)
        fake_names = ["Alexander Wright", "Jonathan Miller", "David K. Harrison", "Robert Vance"]
        new_name = random.choice(fake_names)
        draw_f.text((x, y), new_name, fill="#050505")
        draw_m.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=255)
        
    elif fraud_type == "id_duplication":
        x, y, w, h = coords["reg_id"]
        draw_f.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=bg_patch)
        new_id = f"23BAI{random.randint(9000, 9999)}"
        draw_f.text((x, y), new_id, fill="#000000")
        draw_m.rectangle([x - 5, y - 2, x + w + 5, y + h + 2], fill=255)
        
    elif fraud_type == "photo_swap":
        # Tamper the seal / authentication stamp region
        cx, cy, r = width - 130, 920, 50
        box = [cx - r, cy - r, cx + r, cy + r]
        # Crop an altered seal from another location or clone it with displacement
        seal_crop = forged_img.crop(box)
        # Invert or shift seal
        seal_np = np.array(seal_crop)
        seal_np = np.roll(seal_np, 15, axis=1) # Splicing shift
        forged_img.paste(Image.fromarray(seal_np), (box[0], box[1]))
        draw_m.rectangle(box, fill=255)
        
    elif fraud_type == "grade_modification":
        # Pick one course grade and modify to "A+"
        if coords["grades"]:
            target_grade = random.choice(coords["grades"])
            x, y, w, h, _ = target_grade
            draw_f.rectangle([x - 4, y - 2, x + w + 4, y + h + 2], fill=bg_patch)
            draw_f.text((x + 5, y), "A+", fill="#000000")
            draw_m.rectangle([x - 4, y - 2, x + w + 4, y + h + 2], fill=255)
            
    # Apply microscopic boundary blending to forged region to simulate realistic document editing
    mask_np = np.array(mask)
    if mask_np.sum() > 0:
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        mask = Image.fromarray(dilated)
        
    return forged_img, mask

def build_benchmark_dataset(labels_csv_path, output_images_dir, output_masks_dir):
    """Parses labels.csv and builds all 600 documents with zero-leakage metadata."""
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    font_title, font_bold, font_regular, font_small = get_fonts()
    
    # Read labels.csv
    with open(labels_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    names_pool = [
        "Rajat Srivastava", "Aman Verma", "Priya Sharma", "Sneha Patel", "Rahul Gupta",
        "Karan Mehra", "Neha Singh", "Arjun Reddy", "Simran Kaur", "Vikas Nair",
        "Ananya Roy", "Rohan Das", "Pooja Mishra", "Siddharth Jain", "Divya Pillai"
    ]
    dept_pool = [
        "Artificial Intelligence & Machine Learning",
        "Computer Science and Engineering",
        "Information Technology",
        "Data Science and Analytics",
        "Electronics and Communication"
    ]
    courses_pool = [
        ("CS301", "Data Structures & Algorithms", 4),
        ("CS302", "Database Management Systems", 4),
        ("AI303", "Machine Learning Foundations", 4),
        ("CS304", "Operating Systems", 3),
        ("MA305", "Probability & Statistics", 3),
        ("AI306", "Computer Vision & Image Proc", 3)
    ]
    
    metadata_rows = []
    master_templates = {}
    
    print(f"Generating {len(records)} document benchmark images and masks...")
    
    # Step 1: Assign master template IDs to prevent source document leakage
    for i, rec in enumerate(records):
        img_name = rec["image_name"]
        label = rec["label"]
        fraud_type = rec["fraud_type"]
        
        # Link every fake document to a genuine master ID
        if label == "genuine":
            master_id = f"master_{i+1:04d}"
        else:
            # Pair each forged doc with one of the genuine master docs
            master_idx = (i % 300) + 1
            master_id = f"master_{master_idx:04d}"
            
        rec["master_id"] = master_id
        
    # Step 2: Render documents
    for i, rec in enumerate(records):
        img_name = rec["image_name"]
        label = rec["label"]
        fraud_type = rec["fraud_type"]
        master_id = rec["master_id"]
        
        # Build master document metadata if not exists
        if master_id not in master_templates:
            student_name = names_pool[hash(master_id) % len(names_pool)]
            dept = dept_pool[hash(master_id) % len(dept_pool)]
            reg_id = f"23BAI{(hash(master_id) % 8999) + 1000}"
            cgpa = round(6.5 + (hash(master_id) % 250) / 100.0, 2)
            attendance = 72 + (hash(master_id) % 22)
            
            # Grades
            c_list = []
            for c_code, c_name, cr in courses_pool:
                gr = ["A", "B+", "A+", "B", "O"][hash(master_id + c_code) % 5]
                c_list.append((c_code, c_name, cr, gr))
                
            meta = {
                "name": student_name,
                "reg_id": reg_id,
                "department": dept,
                "batch": "2023 - 2027",
                "semester": "Semester IV",
                "courses": c_list,
                "cgpa": cgpa,
                "attendance": attendance
            }
            auth_img, coords = render_authentic_document(meta, font_title, font_bold, font_regular, font_small)
            master_templates[master_id] = (meta, auth_img, coords)
        else:
            meta, auth_img, coords = master_templates[master_id]
            
        img_path = os.path.join(output_images_dir, img_name)
        mask_name = img_name.replace(".png", "_mask.png")
        mask_path = os.path.join(output_masks_dir, mask_name)
        
        if label == "genuine":
            # Save authentic document and all-zero mask
            auth_img.save(img_path)
            mask_zero = Image.new("L", auth_img.size, 0)
            mask_zero.save(mask_path)
        else:
            # Apply targeted forgery
            forged_img, mask_img = apply_forgery(auth_img, coords, fraud_type, meta, font_bold, font_regular)
            forged_img.save(img_path)
            mask_img.save(mask_path)
            
        metadata_rows.append({
            "image_name": img_name,
            "image_path": img_path,
            "mask_path": mask_path,
            "label": label,
            "fraud_type": fraud_type,
            "master_id": master_id
        })
        
    print(f"Successfully rendered {len(metadata_rows)} images and ground-truth masks.")
    return metadata_rows

def create_leakage_free_splits(metadata_rows, splits_dir="data/splits", seed=42):
    """
    Partitions dataset by master_id using GroupShuffleSplit.
    Ensures zero master template leakage between Train (70%), Val (15%), Test (15%).
    """
    os.makedirs(splits_dir, exist_ok=True)
    
    unique_masters = list(set(r["master_id"] for r in metadata_rows))
    unique_masters.sort()
    random.Random(seed).shuffle(unique_masters)
    
    n_total = len(unique_masters)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_masters = set(unique_masters[:n_train])
    val_masters = set(unique_masters[n_train:n_train + n_val])
    test_masters = set(unique_masters[n_train + n_val:])
    
    train_rows = [r for r in metadata_rows if r["master_id"] in train_masters]
    val_rows = [r for r in metadata_rows if r["master_id"] in val_masters]
    test_rows = [r for r in metadata_rows if r["master_id"] in test_masters]
    
    def write_csv(filepath, rows):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            
    train_file = os.path.join(splits_dir, "train.csv")
    val_file = os.path.join(splits_dir, "val.csv")
    test_file = os.path.join(splits_dir, "test.csv")
    master_file = "data/metadata.csv"
    
    # Add split tag
    for r in metadata_rows:
        if r["master_id"] in train_masters:
            r["split"] = "train"
        elif r["master_id"] in val_masters:
            r["split"] = "val"
        else:
            r["split"] = "test"
            
    write_csv(train_file, train_rows)
    write_csv(val_file, val_rows)
    write_csv(test_file, test_rows)
    write_csv(master_file, metadata_rows)
    
    print("\n--- Leakage-Safe Grouped Dataset Splits ---")
    print(f"Train Set: {len(train_rows)} samples across {len(train_masters)} master templates")
    print(f"Val Set:   {len(val_rows)} samples across {len(val_masters)} master templates")
    print(f"Test Set:  {len(test_rows)} samples across {len(test_masters)} master templates")
    print(f"Saved splits to {splits_dir}/")

def main():
    labels_csv = "Machine-Vision-project-main/labels.csv"
    images_dir = "data/processed/images"
    masks_dir = "data/processed/masks"
    splits_dir = "data/splits"
    
    rows = build_benchmark_dataset(labels_csv, images_dir, masks_dir)
    create_leakage_free_splits(rows, splits_dir)

if __name__ == "__main__":
    main()
