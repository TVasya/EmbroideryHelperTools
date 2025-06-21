import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox, Menu
from tkinter.colorchooser import askcolor
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
from customtkinter import CTkImage

# Constants
DEFAULT_RADIUS = 4
MAX_RADIUS = 50
DEFAULT_THRESHOLD = 99  # percent
DEFAULT_OVERLAY = 0     # percent
DEFAULT_SCALE = 100     # percent
MAX_SCALE = 400         # percent


def binarize(pil_img: Image.Image, threshold: int = 128) -> np.ndarray:
    gray = np.array(pil_img.convert("L"))
    return (gray > threshold).astype(np.uint8)


def majority_filter_circle(binary: np.ndarray, radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = (x**2 + y**2 <= radius**2).astype(np.uint8)
    sum_vals = cv2.filter2D(binary, cv2.CV_32S, kernel, borderType=cv2.BORDER_CONSTANT)
    count_vals = cv2.filter2D(
        np.ones_like(binary, dtype=np.uint8),
        cv2.CV_32S,
        kernel,
        borderType=cv2.BORDER_CONSTANT
    )
    return (sum_vals > (count_vals / 2)).astype(np.uint8)


class MajorityFilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Embroidery Cleaner")
        self.geometry("900x700")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.orig_pil = None
        self.last_processed_save = None
        self.overlay_color = "#FF0000"

        # Build UI
        self._build_menubar()
        self._build_controls()
        self._build_image_area()

        # Control elements for locking
        self._controls = [
            self.thr_slider, self.rad_slider,
            self.ovr_slider, self.ovr_switch,
            self.color_btn, self.inv_switch,
            self.scale_slider, self.remove_small_switch,
            self.min_area_slider
        ]


    def _build_menubar(self):
        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self._load_image_dialog)
        file_menu.add_command(label="Save", command=self._save_image_dialog)
        file_menu.add_command(label="Batch Process...", command=self._batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_controls(self):
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(side="left", fill="y", padx=15, pady=15)

        # Threshold
        ctk.CTkLabel(ctrl, text="Threshold (LDR):").pack(pady=(0,2))
        self.thr_slider = ctk.CTkSlider(ctrl, from_=0, to=99, number_of_steps=100,
                                        command=self._on_change)
        self.thr_slider.set(DEFAULT_THRESHOLD)
        self.thr_slider.pack(fill="x")
        self.thr_value = ctk.CTkLabel(ctrl, text=f"{DEFAULT_THRESHOLD}%")
        self.thr_value.pack(pady=(0,6))

        # Radius
        ctk.CTkLabel(ctrl, text="Circle Radius:").pack(pady=(0,2))
        self.rad_slider = ctk.CTkSlider(ctrl, from_=4, to=MAX_RADIUS, number_of_steps=MAX_RADIUS,
                                        command=self._on_change)
        self.rad_slider.set(DEFAULT_RADIUS)
        self.rad_slider.pack(fill="x")
        self.rad_value = ctk.CTkLabel(ctrl, text=f"{DEFAULT_RADIUS} px")
        self.rad_value.pack(pady=(0,6))

        # Overlay
        ctk.CTkLabel(ctrl, text="Overlay:").pack(pady=(0,2))
        self.ovr_slider = ctk.CTkSlider(ctrl, from_=0, to=100, number_of_steps=100,
                                        command=self._on_change)
        self.ovr_slider.set(DEFAULT_OVERLAY)
        self.ovr_slider.pack(fill="x")
        self.ovr_value = ctk.CTkLabel(ctrl, text=f"{DEFAULT_OVERLAY}%")
        self.ovr_value.pack(pady=(0,6))
        self.ovr_switch = ctk.CTkSwitch(ctrl, text="Enable Overlay (Viewport Only)",
                                        command=self._on_change)
        self.ovr_switch.select()
        self.ovr_switch.pack(pady=(0,6))
        self.color_btn = ctk.CTkButton(ctrl, text="Overlay Color",
                                       command=self._pick_color)
        self.color_btn.pack(pady=(0,10))

        # Invert
        self.inv_switch = ctk.CTkSwitch(ctrl, text="Invert Shapes",
                                        command=self._on_change)
        self.inv_switch.pack(pady=(0,10))

        # Scale
        ctk.CTkLabel(ctrl, text="Scale Factor:").pack(pady=(0,2))
        self.scale_slider = ctk.CTkSlider(ctrl, from_=DEFAULT_SCALE, to=MAX_SCALE,
                                          number_of_steps=MAX_SCALE-DEFAULT_SCALE,
                                          command=self._on_change)
        self.scale_slider.set(DEFAULT_SCALE)
        self.scale_slider.pack(fill="x")
        self.scale_value = ctk.CTkLabel(ctrl, text=f"{DEFAULT_SCALE}%")
        self.scale_value.pack(pady=(0,6))
        
        # Scale Textbox
        scale_text_frame = ctk.CTkFrame(ctrl)
        scale_text_frame.pack(fill="x", pady=(0,6))

        ctk.CTkLabel(scale_text_frame, text="or enter value:").pack(side="left", padx=(0,5))

        self.scale_entry = ctk.CTkEntry(scale_text_frame, width=60)
        self.scale_entry.insert(0, str(DEFAULT_SCALE))
        self.scale_entry.pack(side="left")
        self.scale_entry.bind("<Return>", self._on_scale_entry)

        # Remove small shapes

        self.remove_small_switch = ctk.CTkSwitch(ctrl, text="Remove Small Shapes",
                                                 command=self._on_change)
        self.remove_small_switch.deselect()
        self.remove_small_switch.pack(pady=(0,6))
        ctk.CTkLabel(ctrl, text="Small Shapes Threshold:").pack(pady=(10,2))

        self.min_area_slider = ctk.CTkSlider(ctrl, from_=1, to=1000, number_of_steps=999,
                                             command=self._on_change)
        self.min_area_slider.set(50)
        self.min_area_slider.pack(fill="x")
        self.min_area_value = ctk.CTkLabel(ctrl, text="50 px")
        self.min_area_value.pack(pady=(0,6))



    def _build_image_area(self):
        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(side="right", expand=True, fill="both", padx=15, pady=15)
        self._ctk_img_ref = None

    def _pick_color(self):
        color = askcolor(title="Overlay Color", initialcolor=self.overlay_color)[1]
        if color:
            self.overlay_color = color
            if self.orig_pil:
                self._process_image()

    def _load_image_dialog(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp"),("All","*")]
        )
        if not path:
            return
        try:
            self.orig_pil = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return
        self._process_image()

    def _save_image_dialog(self):
        if self.last_processed_save is None:
            messagebox.showwarning("Warning","No image to save.")
            return
        # Save in original size
        save_img = self.last_processed_save.resize(self.orig_pil.size, resample=Image.NEAREST)
        invert = self.inv_switch.get()
        if invert:
            save_img = ImageOps.invert(save_img)
        bg = (255,255,255) if not invert else (0,0,0)
        rgba = save_img.convert("RGBA")
        data = [(px[0],px[1],px[2],0) if px[:3]==bg else (px[0],px[1],px[2],255)
                 for px in rgba.getdata()]
        rgba.putdata(data)
        out = filedialog.asksaveasfilename(
            title="Save as PNG", defaultextension=".png",
            filetypes=[("PNG","*.png")]
        )
        if not out:
            return
        try:
            rgba.save(out,"PNG")
            messagebox.showinfo("Saved",f"File saved: {out}")
        except Exception as e:
            messagebox.showerror("Save Error",str(e))

    def _batch_process(self):
        in_dir = filedialog.askdirectory(title="Input Folder")
        if not in_dir: return
        out_dir = filedialog.askdirectory(title="Output Folder")
        if not out_dir: return
        thr=int(self.thr_slider.get()*255/100)
        rad=int(self.rad_slider.get())
        alpha=self.ovr_slider.get()/100.0
        use_ovr=self.ovr_switch.get()
        inv=self.inv_switch.get()
        color=self.overlay_color
        scale_pct=self.scale_slider.get()
        threading.Thread(target=self._run_batch,
                         args=(in_dir,out_dir,thr,rad,alpha,use_ovr,inv,color,scale_pct),
                         daemon=True).start()

    def _run_batch(self,in_dir,out_dir,thr,rad,alpha,use_ovr,inv,color,scale_pct):
        exts=('.png','.jpg','.jpeg','.bmp')
        files=[f for f in os.listdir(in_dir) if f.lower().endswith(exts)]
        scale=scale_pct/100.0
        for f in files:
            try:
                img0=Image.open(os.path.join(in_dir,f))
                img=img0.resize((int(img0.width*scale),int(img0.height*scale)),resample=Image.LANCZOS)
                bin_arr=binarize(img,thr)
                filt=majority_filter_circle(bin_arr,rad)
                proc=Image.fromarray((filt*255).astype(np.uint8))
                base=ImageOps.invert(proc) if inv else proc
                # scale back before saving
                base = base.resize(img0.size, resample=Image.LANCZOS)
                save_img=base.convert("RGB")
                bg=(255,255,255) if not inv else (0,0,0)
                rgba=save_img.convert("RGBA")
                data=[(px[0],px[1],px[2],0) if px[:3]==bg else (px[0],px[1],px[2],255)
                      for px in rgba.getdata()]
                rgba.putdata(data)
                rgba.save(os.path.join(out_dir,os.path.splitext(f)[0]+'.png'),'PNG')
            except Exception as e:
                print(f"Error processing {f}: {e}")
        messagebox.showinfo("Complete",f"Processed {len(files)} files.")
    
    def _on_scale_entry(self, event=None):
        try:
            value = int(self.scale_entry.get())
            if DEFAULT_SCALE <= value <= MAX_SCALE:
                self.scale_slider.set(value)
                self._on_change()
            else:
                messagebox.showwarning("Error", f"Enter a value between {DEFAULT_SCALE} and {MAX_SCALE}")
        except ValueError:
            messagebox.showerror("Error", "Enter a whole number")


    def _on_change(self, _=None):
        self.thr_value.configure(text=f"{int(self.thr_slider.get())}%")
        self.rad_value.configure(text=f"{int(self.rad_slider.get())} px")
        self.ovr_value.configure(text=f"{int(self.ovr_slider.get())}%")
        self.scale_value.configure(text=f"{int(self.scale_slider.get())}%")
        self.min_area_value.configure(text=f"{int(self.min_area_slider.get())} px")
        if self.orig_pil:
            self._process_image()


    def _process_image(self):
        thr=int(self.thr_slider.get()*255/100)
        rad=int(self.rad_slider.get())
        alpha=self.ovr_slider.get()/100.0
        use_ovr=self.ovr_switch.get()
        inv=self.inv_switch.get()
        scale=self.scale_slider.get()/100.0
        for w in self._controls: w.configure(state="disabled")
        threading.Thread(target=self._filter_thread,
                         args=(thr,rad,alpha,use_ovr,inv,scale),daemon=True).start()

    def _filter_thread(self,thr,rad,alpha,use_ovr,inv,scale):
        try:
            scaled=self.orig_pil.resize((int(self.orig_pil.width*scale),
                                         int(self.orig_pil.height*scale)),resample=Image.LANCZOS)
            bin_arr=binarize(scaled,thr)
            filt=majority_filter_circle(bin_arr, rad)
            # Remove small shapes (by inverted mask)
            if self.remove_small_switch.get():
                min_area = int(self.min_area_slider.get())
                mask = (1 - filt).astype(np.uint8)  # Invert mask to find filled shapes

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

                new_mask = np.zeros_like(mask)
                for lbl in range(1, num_labels):
                    if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                        new_mask[labels == lbl] = 1

                filt = 1 - new_mask  # Invert back to restore normal view



            proc=Image.fromarray((filt*255).astype(np.uint8))
            self.last_processed_save=ImageOps.invert(proc) if inv else proc
            # Create preview in original size
            preview=self.last_processed_save.convert("RGB").resize(
                (self.orig_pil.width,self.orig_pil.height),resample=Image.LANCZOS)
            if use_ovr:
                tinted=ImageOps.colorize(scaled.convert("L"),black="black",white=self.overlay_color)
                preview=Image.blend(preview,tinted.resize(preview.size,resample=Image.LANCZOS),alpha)
        except Exception as e:
            self.after(0,lambda:messagebox.showerror("Error",str(e)))
            return
        ctk_img=CTkImage(light_image=preview,dark_image=preview,size=preview.size)
        self.after(0,lambda:self._update_image(ctk_img))
        self.after(0,lambda:[w.configure(state="normal") for w in self._controls])

    def _update_image(self, img):
        # Determine window size for display
        max_width = self.winfo_width() - 200  # account for left panel
        max_height = self.winfo_height() - 50

        img_width, img_height = img._light_image.size  # original image size

        # Calculate scale
        scale = min(max_width / img_width, max_height / img_height, 1.0)

        # Scale if image is larger than window
        if scale < 1.0:
            new_size = (int(img_width * scale), int(img_height * scale))
            resized = img._light_image.resize(new_size, resample=Image.LANCZOS)
            ctk_img = CTkImage(light_image=resized, dark_image=resized, size=new_size)
        else:
            ctk_img = img

        self.image_label.configure(image=ctk_img, text="")
        self._ctk_img_ref = ctk_img



if __name__=="__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app=MajorityFilterApp()
    app.mainloop()