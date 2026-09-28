import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="SandGAN Studio", page_icon="🏜️", layout="wide")

# 2. CUSTOM BACKGROUND IMAGE (CSS Magic)
# 2. CUSTOM BACKGROUND IMAGE (Gloomy Cinematic Vignette & Glow)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    /* Gloomy, cinematic spotlight effect (No images, pure aesthetic CSS) */
    background: radial-gradient(circle at 50% 30%, rgba(78, 52, 46, 0.6) 0%, rgba(20, 15, 13, 0.95) 60%, rgba(0, 0, 0, 1) 100%);
    background-color: #050505;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
/* Upload box ko Glassmorphism (premium sheesha) look diya hai */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px !important;
}
/* Text ko eye-catching banane ke liye ek soft glowing effect */
h1, h3 {
    color: #FFCC80 !important; 
    text-shadow: 0px 0px 15px rgba(255, 204, 128, 0.4) !important;
    font-family: 'Segoe UI', sans-serif;
}
p {
    color: #D7CCC8 !important;
    font-family: 'Segoe UI', sans-serif;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# 3. THE AI BLUEPRINT (U-Net Architecture)
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.d1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.d2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.d3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.d4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.d5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        
        self.u1 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.u2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.u3 = nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.u4 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.u5 = nn.Sequential(nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        
        u1 = self.u1(d5)
        u1 = torch.cat([u1, d4], dim=1)
        u2 = self.u2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.u3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.u4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u5 = self.u5(u4)
        return u5

# 4. MODEL LOADING
@st.cache_resource
def load_model():
    model = UNetGenerator()
    model.load_state_dict(torch.load("sandGAN_generator_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

generator = load_model()

# 5. STREAMLIT WEB APP UI (Clean without Sidebar)
st.markdown("<h1 style='text-align: center; color: #FFA07A; text-shadow: 2px 2px 4px #000000;'>🏜️ SandGAN: Sandstorm Image Cleaner</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #F5F5F5; text-shadow: 1px 1px 2px #000000;'>Upload your sandstorm-degraded photo and let Deep Learning restore the true colors!</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# File Uploader ko center mein rakhne ke liye columns ka use kiya hai
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    uploaded_file = st.file_uploader("📂 Upload an Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 6. IMAGE PREPROCESSING
    image = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner("✨ Neural Network is clearing the dust..."):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # 7. AI INFERENCE
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # 8. POST-PROCESSING
        output_tensor = output_tensor.squeeze().cpu()
        output_tensor = (output_tensor + 1) / 2.0
        
        output_img = transforms.ToPILImage()(output_tensor)
        output_img = output_img.resize(image.size)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 9. RESULTS DISPLAY (No Sidebar, just pure results)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("<h3 style='text-align: center; color: white; text-shadow: 1px 1px 2px black;'> Sandstorm </h3>", unsafe_allow_html=True)
            st.image(image, width="stretch")
            
        with col2:
            st.markdown("<h3 style='text-align: center; color: white; text-shadow: 1px 1px 2px black;'> Sand-UNet Cleaned </h3>", unsafe_allow_html=True)
            st.image(output_img,width="stretch")
            
        st.success("Congo! Your photo is clean")