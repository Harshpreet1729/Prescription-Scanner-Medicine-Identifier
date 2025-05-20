# ======= gui_app.py =======
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
from predict import predict
from utils import get_generic_name, get_uses
import threading

def browse_file():
    path = filedialog.askopenfilename()
    if not path:
        return
    image = Image.open(path).resize((250, 250))
    img_tk = ImageTk.PhotoImage(image)
    panel.config(image=img_tk)
    panel.image = img_tk
    result_label.config(text="⏳ Predicting, please wait...")

    def run_prediction():
        try:
            medicine = predict(path)
            generic = get_generic_name(medicine)
            uses = get_uses(generic)
            result = f"🔹 Medicine: {medicine}\n🔹 Generic: {generic}\n🔹 Uses: {uses}"
        except Exception as e:
            result = f"❌ Error: {str(e)}"
        result_label.config(text=result)

    threading.Thread(target=run_prediction).start()

def clear_display():
    panel.config(image='')
    panel.image = None
    result_label.config(text="Upload a prescription image to get started.")

# Setup GUI
app = tk.Tk()
app.title("🩺 AI Prescription Reader")
app.geometry("500x500")
app.configure(bg="#f0f0f0")

tk.Label(app, text="Upload Doctor's Handwritten Prescription", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)
tk.Button(app, text="📤 Upload Prescription", command=browse_file, font=("Arial", 12)).pack(pady=5)

panel = tk.Label(app, bg="#ffffff", relief=tk.SUNKEN, width=250, height=250)
panel.pack(pady=10)

result_label = tk.Label(app, text="Upload a prescription image to get started.", wraplength=450, justify="left", font=("Arial", 11), bg="#f0f0f0")
result_label.pack(pady=10)

tk.Button(app, text="🔄 Clear", command=clear_display, font=("Arial", 11)).pack(pady=5)

app.mainloop()