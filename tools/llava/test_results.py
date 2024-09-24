from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, AutoProcessor, LlavaForConditionalGeneration
from xtuner._lite.modelings import register_remote_code
from xtuner._lite.chat import ChatMessages, CHAT_TEMPLATE_MAP
import torch
from PIL import Image
import os
import requests
def single_image_qa():
    model_path = "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-20B_llava_caption_pretrain_0822_projector/20240911150811/hf-13000-of-13490/"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    model.cuda()
    def test(question, img_path=None):
        if img_path is not None:
            try:
                image = Image.open(img_path)
                question = "<image>\n" + question
            except Exception as e:
                print(f"Error loading image: {e}")
                return        
        msg = ChatMessages.from_str(question)
        prompt = msg.get_prompt(CHAT_TEMPLATE_MAP['internlm2'])
        if img_path is not None:
            inputs = processor(text=prompt, images=image, return_tensors='pt')
        outputs = model.generate(
            input_ids=inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            pixel_values=inputs['pixel_values'].cuda(),
            max_new_tokens=1024)
        response = processor.batch_decode(outputs)[0]
        print("<|im_end|>".join(response.split("<|im_end|>")[:2]))
    # while True:
    #     img_path = input("Enter the image path (or 'exit' to quit): ")
    #     if img_path.lower() == 'exit':
    #         break
    #     question = input("Enter the question for the image: ")
    img_path_list=[ 
        # None,
        # "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_1.jpg",
        # "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_2.jpg",
        "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_3.jpg",
        # "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_4.png",
        # "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_5.png",
        # "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_6.png",
        # "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_7.png",
        "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_7.png",
        ]
    question_list = [ 
        # """Assessing the boxed area in this endoscopy image, which artifact stands out?\ nA. blood artifacts\n        B. bubbles\n        C. low contrast\n        D. instrument artifacts\n        E. blood artifacts""",        
        # """View the Endoscopy image provided. What organ is displayed in the marked area?\n        A. suturing needle\n        B. instrument suction\n        C. small intestine\n        D. instrument clasper\n    E. clamps""", 
        # """What object does the marked area in a given CT image show?""",
        """Answer the following question and explain your reasons.
        Question: Which of the following options best matches the symptoms of marked region shown in the CT image?
        A. enhancing tumor
        B. liver tumor
        C. colon cancer primaries
        D. lung nodule
        E. brain aneurysm
        """,
        # """请为该图像生成详细的诊断报告""",
        # """分析图像中展现的疾病""",
        # """分析图像中展现的疾病""",
        # """分析左右两张子图中哪张是covid-19,并给出详细解释""",
        """分析左右两张子图中哪张是covid-19,并给出详细解释原因""",
        ]
    for img_path, question in zip(img_path_list, question_list):
        print(img_path)
        test(question, img_path)

def single_turn():
    model_path = "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-llava_caption_pretrain_image-encoder+projector/20240909213053/hf-13490-of-13490"
    if not os.path.exists(os.path.join(model_path, "modeling_llava.py")):
        os.system(f"cp /cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-20B_llava_caption_pretrain_0822_projector/20240911150811/hf-13000-of-13490/modeling_llava.py {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    model.cuda()
    while True:
        img_path = input("Enter the image path (or 'exit' to quit): ")
        if img_path.lower() == 'exit':
            break
        question = input("Enter the question for the image: ")
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        msg = ChatMessages.from_str("<image>\n" + question)
        prompt = msg.get_prompt(CHAT_TEMPLATE_MAP['internlm2'])

        inputs = processor(text=prompt, images=image, return_tensors='pt')
        outputs = model.generate(
            input_ids=inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            pixel_values=inputs['pixel_values'].cuda(),
            max_new_tokens=1024)
        response = processor.batch_decode(outputs)[0]
        print("<|im_end|>".join(response.split("<|im_end|>")[:2]))


# def multi_turn_conversation():
#     model_path = "/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-llava_caption_pretrain_image-encoder+projector/20240909213053/hf-13490-of-13490"
#     if not os.path.exists(os.path.join(model_path, "modeling_llava.py")):
#         os.system(f"cp /cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-20B_llava_caption_pretrain_0822_projector/20240911150811/hf-13000-of-13490/modeling_llava.py {model_path}/")
    
#     processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
#     model.cuda()

#     conversation_history = []
#     image = None

#     while True:
#         if not image:
#             img_path = input("Enter the image path (or 'exit' to quit): ")
#             if img_path.lower() == 'exit':
#                 break
#             try:
#                 image = Image.open(img_path)
#             except Exception as e:
#                 print(f"Error loading image: {e}")
#                 continue

#         question = input("Enter your question (or 'new image' to change image, 'exit' to quit): ")
#         if question.lower() == 'exit':
#             break
#         if question.lower() == 'new image':
#             image = None
#             conversation_history = []
#             continue

#         conversation_history.append(f"Human: {question}")
        
#         # Construct the prompt with the entire conversation history
#         full_prompt = "<image>\n" + "\n".join(conversation_history)
#         msg = ChatMessages.from_str(full_prompt)
#         prompt = msg.get_prompt(CHAT_TEMPLATE_MAP['internlm2'])

#         inputs = processor(text=prompt, images=image, return_tensors='pt')
#         outputs = model.generate(
#             input_ids=inputs['input_ids'].cuda(),
#             attention_mask=inputs['attention_mask'].cuda(),
#             pixel_values=inputs['pixel_values'].cuda(),
#             max_new_tokens=1024)
#         response = processor.batch_decode(outputs)[0]
        
#         # Extract the model's response
#         model_response = "<|im_end|>".join(response.split("<|im_end|>")[:2]).replace("<image>", "").strip()
#         print("Assistant:", model_response)
        
#         # Add the model's response to the conversation history
#         conversation_history.append(f"Assistant: {model_response}")

# multi_turn_conversation()

# # img_path = '/cpfs01/shared/gmai/xtuner_workspace/mimic-cxr-jpg/2.1.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg'
# # question = "please generate a detail report for this x-ray image."
# # model_path = '/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-20B_llava_caption_pretrain_0822_projector/20240911150811/hf-13000-of-13490/'
# # model_path = '/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-llava_caption_pretrain_0822_projector/20240905163202/hf-12854-of-12854'

# def multi_turn():
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
model = '/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/work_dir/internlm2_5-llava_caption_pretrain_image-encoder+projector/20240909213053/hf-13490-of-13490'
chat_template_config = ChatTemplateConfig('internvl-internlm2')
pipe = pipeline(model, chat_template_config=chat_template_config,
                backend_config=TurbomindEngineConfig(session_len=8192))
image = load_image('/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/tools/test_images/test_10.png')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.9)
sess = pipe.chat(('这张图像上是否存在DRUSEN、CNV或者DME，(好像是DME)', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('请更加详细解释什么是DME', session=sess, gen_config=gen_config)
print(sess.response.text)
# single_image_qa()