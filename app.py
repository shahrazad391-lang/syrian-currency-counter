import gradio as gr
from ultralytics import YOLO
from PIL import Image
import re

# تحميل النموذج
model = YOLO("best.pt")

def parse_details_to_r(details_text):
    if not details_text or "لم يتم" in details_text:
        return 0
    total = 0
    # إصلاح مشكلة الجمع: استخراج الأرقام من النص حتى لو كان يحتوي على كلمات
    for line in details_text.split("\n"):
        nums = re.findall(r'\d+', line)
        if nums:
            num = int(nums[0])
            if "new" in line.lower() or "جديد" in line.lower():
                total += num * 100
            else:
                total += num
    return total

def detect(image):
    if image is None:
        return None, "لم يتم الكشف", 0, None
    
    results = model(image, conf=0.35)
    annotated = Image.fromarray(results[0].plot())
    details = [f"• {model.names[int(box.cls)]}" for box in results[0].boxes]
    details_text = "\n".join(details) if details else "لم يتم الكشف"
    
    # حساب R تلقائياً وعرضه للمستخدم ليعرف على ماذا سيوافق
    r_val = parse_details_to_r(details_text)
    
    # إرجاع None للراديو لإلغاء التحديد الافتراضي
    return annotated, details_text, r_val, None

css = """
.gradio-container {background-color: #f8f9fa;}
.header {font-size: 34px; font-weight: bold; color: #1b5e20; text-align: center;}
.section-a {border: 6px double #1b5e20; border-radius: 25px; padding: 20px; background: linear-gradient(to bottom, #e8f5e9, #c8e6c9);}
.section-b {border: 5px solid #2e7d32; border-radius: 20px; background: #f0f8e8; padding: 18px;}
.arrow {font-size: 32px; color: #2e7d32; text-align: center; margin: 12px 0;}
.gr-number input {font-size: 1.9rem !important; font-weight: bold;}
"""

with gr.Blocks(css=css, title="العملة السورية") as demo:
    
    gr.Markdown("""
    <div class="header">
        🐙💵 العملة السورية<br>
        <span style="font-size:24px;">Syrian Currency Counter</span>
    </div>
    """)

    # متغير لتخزين المجموع الكلي التراكمي
    total_old_state = gr.State(0)

    with gr.Column(elem_classes="section-a"):
        gr.Markdown("**A — ارفع الصورة هنا أو التقطها**")
        img_input = gr.Image(type="pil", label="Upload or take photo", sources=["upload", "webcam"], height=320)
        detect_btn = gr.Button("🔍 كشف / Detect", variant="primary", size="lg")

    with gr.Column(elem_classes="section-b"):
        gr.Markdown("**B — الافتراضي بعد الكشف (وجهة نظر الأخطبوط 🐙)**")
        annotated_out = gr.Image(height=280)
        details_out = gr.Textbox(label="التفاصيل / Detected Details", lines=5)

    gr.Markdown("**هل هذا صحيح؟ is This correct?**")
    confirm = gr.Radio(["✅ نعم", "❌ لا"], label="") # تمت إزالة القيمة الافتراضية

    gr.Markdown("""
    <div class="arrow">↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓</div>
    """)

    with gr.Row():
        r_box = gr.Number(label="R 🐙 — مجموع الصورة (يُحسب تلقائياً)", value=0, interactive=False)
        p_input = gr.Number(label="P — إدخال يدوي إذا اخترت 'لا' (بالقديم)", value=0)

    with gr.Column(elem_classes="section-b"):
        gr.Markdown("**هل تريد إضافة مبلغ كتابة؟**")
        with gr.Row():
            manual_h = gr.Number(label="H - قديم", value=0, precision=0)
            gr.Markdown("**+**")
            manual_s = gr.Number(label="S - جديد", value=0, precision=0)

    add_btn = gr.Button("✅ إضافة إلى المجموع الكلي", variant="primary", size="lg")

    gr.Markdown("**T — المجموع الكلي 🟩⬜⬛⭐⭐⭐**") 
    with gr.Row():
        total_old_box = gr.Number(label="Total Old (ل.س)", value=0, interactive=False)
        total_new_box = gr.Number(label="Total New (÷100)", value=0, interactive=False)

    reset_btn = gr.Button("🔄 Reset All / البَدء من جديد", variant="stop", size="lg")

    # --- الدوال المساعدة ---
    
    def add_to_total(conf, r_val, p_val, h, s, current_total):
        if not conf:
            raise gr.Error("يرجى اختيار '✅ نعم' أو '❌ لا' قبل الإضافة!")
            
        # تحديد القيمة الأساسية بناءً على الاختيار
        if conf == "✅ نعم":
            base_val = float(r_val or 0)
        else:
            base_val = float(p_val or 0)
            
        # حساب الإضافة الجديدة حسب معادلتك تماماً
        added = base_val + float(h or 0) + (float(s or 0) * 100)
        
        # المجموع الجديد
        new_total_old = current_total + added
        new_total_new = new_total_old / 100
        
        # إرجاع: [تحديث المتغير التراكمي، عرض المجموع القديم، عرض المجموع الجديد، تصفير P، تصفير H، تصفير S]
        return new_total_old, new_total_old, new_total_new, 0, 0, 0

    def reset_all():
        return 0, 0, 0, None, "", None, 0, 0, 0

    # --- ربط الأحداث ---
    
    # عند الضغط على كشف
    detect_btn.click(
        detect, 
        inputs=img_input, 
        outputs=[annotated_out, details_out, r_box, confirm]
    )

    # عند الضغط على إضافة
    add_btn.click(
        add_to_total,
        inputs=[confirm, r_box, p_input, manual_h, manual_s, total_old_state],
        outputs=[total_old_state, total_old_box, total_new_box, p_input, manual_h, manual_s]
    )

    # عند الضغط على إعادة تعيين
    reset_btn.click(
        reset_all, 
        outputs=[total_old_state, total_old_box, total_new_box, confirm, details_out, annotated_out, r_box, manual_h, manual_s]
    )

demo.launch()
