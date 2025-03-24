from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from summarizer.sbert import SBertSummarizer
from pprint import pprint

def get_patient_health_records():
    """Retrieves a list of simulated patient health records."""
    return [
        "Patient presents with a persistent dry cough, no fever. Chest X-ray reveals mild hyperinflation. Suspect possible allergies.",
        "Patient reports shortness of breath and chest tightness, especially after exertion. Auscultation reveals wheezing. Diagnosed with asthma exacerbation.",
        "Patient presents with high fever, chills, and productive cough (green phlegm). Chest X-ray shows consolidation in the right lower lobe. Diagnosed with pneumonia.",
        "Patient reports chest pain, radiating to the left arm. EKG shows no acute changes. Suspect musculoskeletal pain.",
        "Patient presents with hemoptysis (coughing up blood). Chest CT scan reveals a small pulmonary nodule. Further investigation needed.",
        "Patient reports pain and swelling in the right knee after a fall. X-ray shows no fracture. Suspect soft tissue injury.",
        "Patient presents with chronic ankle pain and instability. MRI reveals ligament tear. Recommend physical therapy.",
        "Patient reports burning sensation and numbness in the feet. Suspect peripheral neuropathy. Further neurological evaluation recommended.",
        "Patient presents with a painful bunion on the left foot. Recommend conservative management initially.",
        "Patient reports calf pain and swelling. Doppler ultrasound reveals deep vein thrombosis (DVT). Requires anticoagulation.",
        "Patient reports shoulder pain and limited range of motion. MRI reveals rotator cuff tear. Recommend arthroscopic surgery.",
        "Patient presents with elbow pain after overuse. Diagnosed with lateral epicondylitis (tennis elbow).",
        "Patient reports numbness and tingling in the fingers. Suspect carpal tunnel syndrome. Nerve conduction studies recommended.",
        "Patient presents with a wrist fracture after a fall. Requires casting.",
        "Patient reports muscle weakness in the arm. Neurological examination reveals possible nerve impingement.",
        "Patient reports severe headache and stiff neck. Lumbar puncture performed, ruling out meningitis.",
        "Patient presents with abdominal pain and vomiting. CT scan reveals appendicitis. Requires surgery.",
        "Patient reports fatigue and weight loss. Further blood tests and imaging studies needed to determine the cause.",
        "Patient presents with a skin rash. Biopsy performed to determine the diagnosis.",
        "Patient reports anxiety and insomnia. Recommend cognitive behavioral therapy and lifestyle changes.",
        "Patient presents with high fever, chills, and productive cough (green phlegm). Chest X-ray shows consolidation in the right lower lobe. Diagnosed with pneumonia.",
        "Patient presents with a persistent cough, shortness of breath, and wheezing. Oxygen saturation is 90% on room air. Chest X-ray shows hyperinflation and flattened diaphragm. Diagnosed with COPD exacerbation.",
        "Patient reports sharp chest pain that worsens with deep breathing. Auscultation reveals pleural rub. Diagnosed with pleurisy.",
        "Patient presents with fever, cough, and night sweats. Chest X-ray shows a cavitary lesion in the upper lobe. Sputum culture positive for Mycobacterium tuberculosis. Diagnosed with pulmonary tuberculosis.",
        "Patient reports sudden onset of shortness of breath and chest pain. CT pulmonary angiogram reveals a pulmonary embolism. Started on anticoagulation therapy.",
        "Patient presents with a history of asthma. Currently experiencing difficulty breathing and using accessory muscles. Peak expiratory flow rate is significantly reduced. Diagnosed with acute severe asthma."
    ]

def run_multimodal_rag(url, probability_threshold, max_values):
    """
    Runs multimodal Retrieval Augmented Generation (RAG) on an X-ray image and patient records.

    Args:
        url: URL of the X-ray image.
        probability_threshold: Minimum probability for relevance.
        max_values: Maximum number of relevant records to retrieve.

    Returns:
        A summary of the relevant patient records.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    patient_records = get_patient_health_records()

    try:
        image = Image.open(requests.get(url=url, stream=True).raw)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    inputs = processor(text=patient_records, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    pprint(probs, width=10)

    top_values, top_indices = torch.topk(probs, max_values)
    top_values = top_values.tolist()[0]
    top_indices = top_indices.tolist()[0]

    relevant_records = [
        patient_records[index]
        for value, index in zip(top_values, top_indices)
        if value > probability_threshold
    ]
    print("Relevant Patient Records:", relevant_records)

    summarizer = SBertSummarizer('paraphrase-MiniLM-L6-v2')
    summary = summarizer(" ".join(relevant_records), num_sentences=5)
    return summary

if __name__ == '__main__':
    image_url = "https://healthimaging.com/sites/default/files/styles/top_stories/public/assets/articles/4996132.jpg.webp?itok=sR1hg4KS"
    probability_threshold = 0.1
    max_values = 5

    summary = run_multimodal_rag(image_url, probability_threshold, max_values)
    if summary:
        print("Summary:", summary)