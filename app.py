
import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract
import cv2
import numpy as np
import ollama


from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss




st.title("🤖 AI PDF Chatbot")

uploaded_file = st.file_uploader(
    "📎 Upload your PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("อัปโหลด PDF สำเร็จ!")

    pdf_bytes = uploaded_file.read()

    images = convert_from_bytes(
    pdf_bytes,
    dpi=300
)
    text = ""


    # OCR
    for i, image in enumerate(images):
        image = np.array(image)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        # ขยายภาพก่อน OCR

        gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
        )

        _, processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        gray = cv2.convertScaleAbs(
         gray,
         alpha=1.5,
         beta=0
        )

        blur = cv2.GaussianBlur(gray, (0, 0), 3)

        sharp = cv2.addWeighted(
         gray,
         1.5,
         blur,
         -0.5,
         0
        )
        
    
        page_text = pytesseract.image_to_string(
             sharp,
             lang="tha",
             config="--oem 3 --psm 3"
             )


        page_text = page_text.replace("WA", "พ.ศ.")
        page_text = page_text.replace("วันที", "วันที่")
        

        text += f"\n--- หน้า {i + 1} ---\n"
        text += page_text

    #Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    st.write("จำนวน Chunks:", len(chunks))

    #Embedding
    model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)
    st.write("chunks type:", type(chunks))
    st.write("chunks length:", len(chunks))
    st.write("first chunk:", chunks[0] if chunks else None)

    embeddings = model.encode(chunks)

    #เก็บ Vector ใน FAISS
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
     
    index.add(
      np.array(embeddings).astype("float32")
)

    st.success("✅ สร้าง Vector Database สำเร็จ")
    st.write("จำนวน Chunks:", len(chunks))
    st.write("ขนาด Vector:", embeddings.shape)

# เรียกใช้ LLM
    with st.spinner("🤖 AI กำลังสรุป PDF..."):
         response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                "role": "user",
                "content": f"""
ช่วยสรุปเอกสารนี้เป็นภาษาไทย 
ใช้ภาษาที่เข้าใจง่าย 
ห้ามแต่งข้อมูลเพิ่ม
หากมีคำภาษาอังกฤษให้แปลเป็นภาษาไทย หากคำนั้นไม่มีความหมายให้แสดงคำศัพท์นั้นเป็นภาษาอังกฤษ
ห้ามแก้ไขหรือเดาตัวเลขจากเอกสาร
ให้คงตัวเลขตามที่ปรากฏในเอกสาร


เอกสาร:
{text}
"""
            }
        ]
    )

    summary = response["message"]["content"]

    st.success("✅ ประมวลผลเสร็จแล้ว")

    st.subheader("📝 สรุปเอกสาร")
    st.write(summary)

# ผู้ใช้พิมพ์คำถามเกี่ยวกับ PDF
    question = st.chat_input("💬 ถามคำถามเกี่ยวกับ PDF")

    if question is not None and question.strip() != "":
     st.write("👤 คุณ:", question)

    # 1. แปลงคำถามเป็น Vector
     question_embedding = model.encode([question])


    # 2. ค้นหา 5 Chunks ที่เกี่ยวข้องที่สุด
     distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=9
    )

    #แสดงผลการค้นหา Chunks ที่เกี่ยวข้อง
     st.subheader("🔎 ผลการค้นหา")

     for rank, (distance, idx) in enumerate(
         zip(distances[0], indices[0])
    ):
         st.write(f"อันดับ {rank + 1}")
         st.write("Distance:", distance)
         st.write("Chunk:", chunks[idx])

    # 3. ดึงข้อความจาก Chunks
     relevant_chunks = [
        chunks[i]
        for i in indices[0]
    ]
   
    # 4. รวม Chunks
     context = "\n\n".join(relevant_chunks)

    # 5. ส่งข้อมูลให้ Llama
     with st.spinner("🤖 AI กำลังค้นหาคำตอบ..."):

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": f"""

ตอบคำถามโดยใช้ข้อมูลจากเอกสารด้านล่างเท่านั้น
ห้ามตอบว่า "ไม่พบข้อมูลในเอกสาร" หากข้อมูลในข้อความสามารถใช้ตอบคำถามได้
ห้ามแต่งข้อมูลเพิ่ม
ห้ามเปลี่ยนแปลงวันที่ ตัวเลข หรือเงื่อนไข
หากเอกสารระบุว่า "วันถัดจากวันประกาศในราชกิจจานุเบกษา"
ให้ตอบตามข้อความนี้โดยตรง



คำถาม:
{question}

ข้อมูลจากเอกสาร:
{context}
"""
                }
            ]
        )

    # 6. แสดงคำตอบ
    answer = response["message"]["content"]

    st.subheader("🤖 คำตอบ")
    st.write(answer)