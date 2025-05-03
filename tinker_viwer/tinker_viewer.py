import os
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET

get_unknown = True
t1_classes_set = set([
    "aeroplane", "bicycle", "bird", "boat", "bus", "car",
    "cat", "cow", "dog", "horse", "motorbike", "sheep", "train",
    "elephant", "bear", "zebra", "giraffe", "truck", "person"
])

# --- Your annotation parsing function here ---
def parse_voc_annotation(xml_path, image_path,image_id):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        instances = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox_xml = obj.find("bndbox")
            xmin = float(bbox_xml.find("xmin").text)
            ymin = float(bbox_xml.find("ymin").text)
            xmax = float(bbox_xml.find("xmax").text)
            ymax = float(bbox_xml.find("ymax").text)

            x = int(xmin)
            y = int(ymin)
            w = int(xmax - xmin)
            h = int(ymax - ymin)
            if get_unknown:
                instances.append({
                    "label": label if label in t1_classes_set else "unknown",
                    "bbox": [x, y, w, h]
                })
            else:
                instances.append({
                    "label": label,
                    "bbox": [x, y, w, h]
                })
    except:
        print("File not found :{xml_path}")
        return []
    if len(instances) > 2:
        print(instances)
        return instances
    return []

def draw_dotted_rectangle(image, x, y, w, h, color, thickness=1, gap=10):
    # Top edge
    for i in range(x, x + w, gap * 2):
        cv2.line(image, (i, y), (min(i + gap, x + w), y), color, thickness)
    # Bottom edge
    for i in range(x, x + w, gap * 2):
        cv2.line(image, (i, y + h), (min(i + gap, x + w), y + h), color, thickness)
    # Left edge
    for i in range(y, y + h, gap * 2):
        cv2.line(image, (x, i), (x, min(i + gap, y + h)), color, thickness)
    # Right edge
    for i in range(y, y + h, gap * 2):
        cv2.line(image, (x + w, i), (x + w, min(i + gap, y + h)), color, thickness)

# --- GUI Class ---
class VOCViewer:
    def __init__(self, master, voc_base , grid_size):
        self.master = master
        self.voc_base = voc_base
        self.grid_size = grid_size  # Number of rows and columns in the grid
        self.num_images = grid_size[0] * grid_size[1]


        # Load image IDs
        val_file = os.path.join("/Users/amiteshgangrade/Desktop/building_neural_net/selected_images.txt") #set your path for image ids
        with open(val_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        print(self.image_ids)
        self.index = 0

        # Setup canvas
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()
        self.canvas_labels = []
        for i in range(self.grid_size[0]):
            row = []
            for j in range(self.grid_size[1]):
                label = tk.Label(self.canvas_frame)
                label.grid(row=i, column=j, padx=5, pady=5)
                row.append(label)
            self.canvas_labels.append(row)

        # Control buttons
        btn_frame = tk.Frame(master)
        btn_frame.pack()

        tk.Button(btn_frame, text="<< Prev", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Next >>", command=self.next_image).pack(side=tk.LEFT)

        self.load_image()

    def load_image(self):
        for i in range(self.num_images):
            idx = (self.index + i) % len(self.image_ids)
            image_id = self.image_ids[idx]
            print(image_id)
            xml_path = os.path.join(self.voc_base, "Annotations", f"{image_id}.xml")
            img_path = os.path.join(self.voc_base, "JPEGImages", f"{image_id}.jpg")
        
            instances = parse_voc_annotation(xml_path, img_path,image_id)
            if len(instances)==0:
                print("No instances found in the image. For Image Id : ",image_id)
                return
            image = cv2.imread(img_path)

        # Draw bounding boxes
            for inst in instances:
                x, y, w, h = inst["bbox"]
                label = inst["label"]
                if label=="unknown":
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 165, 255), 2)
                    # cv2.putText(image, label[0].capitalize()+label[1:], (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 165, 255),1, cv2.LINE_AA)
                    # Calculate text size
                    (text_width, text_height), baseline = cv2.getTextSize(label[0].capitalize() + label[1:], cv2.FONT_HERSHEY_TRIPLEX, 0.6, 1)
                    top_left = (x, y - text_height - 5)  # Top-left corner of the rectangle
                    bottom_right = (x + text_width, y)
                    overlay = image.copy()

                    cv2.rectangle(overlay, top_left, bottom_right, (0, 165, 255), -1)  # Yellow background (BGR: 0, 255, 255)

                    alpha = 0.6  # Transparency factor (0.0 to 1.0, where 1.0 is opaque)
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                    cv2.putText(image, label[0].capitalize() + label[1:], (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 150, 0), 3)
                    #cv2.putText(image, label[0].capitalize()+label[1:], (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 150, 0),1, cv2.LINE_AA)
                    (text_width, text_height), baseline = cv2.getTextSize(label[0].capitalize() + label[1:], cv2.FONT_HERSHEY_TRIPLEX, 0.6, 1)
                    top_left = (x, y - text_height - 5)  # Top-left corner of the rectangle
                    bottom_right = (x + text_width, y)
                    overlay = image.copy()

                    cv2.rectangle(overlay, top_left, bottom_right, (0, 150, 0), -1)  # Yellow background (BGR: 0, 255, 255)

                    alpha = 0.6  # Transparency factor (0.0 to 1.0, where 1.0 is opaque)
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                    cv2.putText(image, label[0].capitalize() + label[1:], (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(image, f"Image ID: {image_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            output_path = os.path.join("/Users/amiteshgangrade/Desktop/image_with_bboxes", f"{image_id}_bbbox.jpg")
            cv2.imwrite(output_path, image)
            print(f"Image with bounding boxes saved to {output_path}")
        # Convert image to displayable format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = image.resize((600, 400))
            photo = ImageTk.PhotoImage(image)

            row, col = divmod(i, self.grid_size[1])
            self.canvas_labels[row][col].configure(image=photo)
            self.canvas_labels[row][col].image = photo  # Save a reference!

    def next_image(self):
        self.index = (self.index + 4) % len(self.image_ids)
        self.load_image()

    def prev_image(self):
        self.index = (self.index - 4) % len(self.image_ids)
        self.load_image()

# --- Run it ---
if __name__ == "__main__":
    print("Running the tinker viewer...")
    voc_base = "/Users/amiteshgangrade/Desktop/building_neural_net/VOCdevkit/VOC2012"  # <- set your path for annotations
    root = tk.Tk()
    root.title("Pascal VOC Viewer")
    app = VOCViewer(root, voc_base , grid_size=(2,2))
    root.mainloop()