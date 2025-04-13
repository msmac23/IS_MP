from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import gradio as gr
import time

# Load model & tokenizer
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a question-answering pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Learning style context
context = (
    "The four types of learning styles: Visual, Auditory, Reading/writing and Kinesthetic. "
    "Describe visual learners - Visual learners prefer visual representations of information, such as diagrams, charts, graphs, and maps. "
    "However, they don't necessarily respond well to photos or videos, rather needing their information using different visual aids such as patterns and shapes. "
    "Describe auditory learners - Auditory learners are individuals who learn through listening. They are prone to sorting their ideas after speaking rather than thinking ideas through before. "
    "Describe Kinesthetic learners - Kinesthetic learners are individuals who prefer to learn by doing and hands-on experience. "
    "Describe reading/writing learners - Reading/writing learners prefer to engage with text-based information. These individuals usually perform very well on written assignments. "
    "To find out your learning style: take the VARK questionnaire linked above, observation of study habits, self-assessment of information retention preferences, experimentation with different study techniques. "
    "Does age affect learning styles: Children often benefit from kinesthetic methods, teenagers from social-learning approaches, adults from reading/writing strategies for professional development."
)

# Predefined study guides
study_guides = {
    'visual': [
        "Use diagrams, charts, and mind maps to organize information",
        "Color-code notes and materials to highlight key concepts",
        "Watch educational videos or animations",
        "Convert text into flowcharts or infographics",
        "Use flashcards with images and symbols",
        "Create visual timelines for historical events",
        "Use graphic organizers to compare/contrast concepts",
        "Utilize color-coded sticky notes for spatial organization"
    ],
    'auditory': [
        "Participate in group discussions and study groups",
        "Use mnemonic devices and rhymes for memorization",
        "Record and listen to audio notes",
        "Explain concepts out loud to yourself",
        "Listen to educational podcasts",
        "Create songs or rhythms for memorization",
        "Use text-to-speech software for reading materials",
        "Engage in debate-style practice sessions"
    ],
    'kinesthetic': [
        "Engage in hands-on activities and experiments",
        "Take active breaks during study sessions",
        "Use physical objects to demonstrate concepts",
        "Incorporate movement into learning routines",
        "Practice through role-playing scenarios",
        "Build 3D models of abstract concepts",
        "Use gesture-based mnemonics",
        "Combine study with light exercise like pacing"
    ],
    'reading/writing': [
        "Write summaries and paraphrased notes",
        "Create detailed outlines and bullet points",
        "Read textbooks and articles thoroughly",
        "Write essays and journal entries",
        "Use written flashcards for review",
        "Convert diagrams into written descriptions",
        "Maintain a detailed glossary of terms",
        "Rewrite notes in multiple formats (lists, paragraphs)"
    ]
}

# Functions
def bot_typing(message, history):
    history.append((message, None))
    time.sleep(1.0)
    QA_input = {'question': message, 'context': context}
    res = nlp(QA_input)
    answer = res['answer']
    history[-1] = (message, answer)
    return history

def generate_guide(style):
    style = style.lower()
    guide = "\n".join([f"{i+1}. {tip}" for i, tip in enumerate(study_guides[style])])
    return f"📚 Custom Study Guide for {style.capitalize()} Learners:\n\n{guide}"
# Add background image URL here
bg_image_url = "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=1350&q=80"

# Update custom CSS for background image
custom_css = f"""
.gradio-container {{
  background-image: url("{bg_image_url}");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
  min-height: 100vh;
  padding: 20px;
}}
.chatbot {{
  background: rgba(249, 249, 249, 0.9);
  border-radius: 16px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}}
.message.user {{
  background-color: rgba(209, 245, 211, 0.9);
  color: black;
  border-radius: 20px;
  padding: 12px;
  max-width: 75%;
  align-self: flex-end;
  backdrop-filter: blur(2px);
}}
.message.bot {{
  background-color: rgba(240, 240, 240, 0.9);
  color: black;
  border-radius: 20px;
  padding: 12px;
  max-width: 75%;
  align-self: flex-start;
  backdrop-filter: blur(2px);
}}
.textbox, .dropdown, .button {{
  background: rgba(255, 255, 255, 0.85) !important;
}}
.accordion {{
  background: rgba(255, 255, 255, 0.8) !important;
}}
"""

# Build the app
with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.HTML("<h1 style='text-align:center; color: #fdfffa;'>🎓 Learning Style Assistant</h1>")
    gr.HTML('<div style="text-align:center; margin-bottom: 20px;"><a href="https://vark-learn.com/the-vark-questionnaire/" target="_blank" style="text-decoration: none; color: #fdfffa;">📝 TAKE THE VARK LEARNING STYLES QUIZ HERE!!!</a></div>')

    chatbot = gr.Chatbot(
        label="Learning Style Q&A Assistant",
        avatar_images=(
            "https://cdn-icons-png.flaticon.com/512/847/847969.png",
            "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
        ),
        bubble_full_width=False,
        height=500
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask me about learning styles...",
            show_label=False,
            container=False
        )
        send_btn = gr.Button("🚀 Send")

    history = gr.State([])

    send_btn.click(bot_typing, inputs=[msg, history], outputs=[chatbot])
    send_btn.click(lambda: "", None, outputs=msg)

    with gr.Accordion("📚 Generate Custom Study Guide", open=False):
        style_dropdown = gr.Dropdown(
            choices=["Visual", "Auditory", "Kinesthetic", "Reading/Writing"],
            label="Select Your Learning Style"
        )
        guide_output = gr.Textbox(label="Your Study Guide", lines=10)
        guide_btn = gr.Button("✨ Generate")
        guide_btn.click(generate_guide, inputs=style_dropdown, outputs=guide_output)

# Launch (keep this the same)
#if __name__ == "__main__":
 #   demo.launch()

demo.launch()
