import os
import gradio as gr
from gradio.components import HTML
import torch
from PIL.ImageOps import scale
from PIL import Image
import json
from zhipuai import ZhipuAI
from bert_Inference import Bert_Inference

# Initialize ZhipuAI client
client = ZhipuAI(api_key="个人APIKEY")
sales_knowledge_id  = "药品销售数据分析结果的知识库ID"
concept_knowledge_id = "药品本身相关的知识库ID"

# Function to classify pharmacy
def classify_commodity(commodity_name):
    category = Bert_Inference(task='class_prediction', model_path='drug_class_prediction_best_model.pth', x=commodity_name)
    # print(category)
    category = category[0]
    return category


# Function to generate sales suggestions
def generate_suggestions(messages, commodity_name, category):
    countries = ['Germany', 'Poland']

    known_facts_dict = {}

    for country in countries:
        # Load the analysis result from the JSON file
        analysis_file = f'data_analysis/analysis_result/Pharm-data/analysis_result_{category}-{country}.json'

        try:
            with open(analysis_file, 'r') as f:
                analysis_result = json.load(f)

            # Extract the relevant information from the analysis result
            known_facts = {
                "product": f"{category} ({country})",
                "sales_summary": analysis_result["sales_summary"],
                "quantity_summary": analysis_result["quantity_summary"],
                "price_summary": analysis_result["price_summary"],
                "top_10_distributor_performance": analysis_result["top_10_distributor_performance"],
                "team_performance": analysis_result["team_performance"],
                "channel_performance": analysis_result["channel_performance"],
                "Pharmacy": analysis_result["channel_performance"]["Pharmacy"],
                "price_sensitivity": analysis_result["price_sensitivity"],
                "sales_rep_distribution": analysis_result["sales_rep_distribution"],
                "average_transaction_value": analysis_result["average_transaction_value"],
                "customer_rfm_analysis": analysis_result["customer_rfm_analysis"],
                "quantity_vs_price_insights": analysis_result["quantity_vs_price_insights"],
                f"top_10_cities_in_{country}_by_sales": analysis_result[
                    f"top_10_cities_in_{country}_by_sales_of_{category}"],
                "top_10_customers_by_sales": analysis_result["top_10_customers_by_sales"],
                "top_10_products_by_sales": analysis_result["top_10_products_by_sales"],
                "sales_regression_analysis": analysis_result["sales_regression_analysis"],
                "clustering_analysis": analysis_result["clustering_analysis"],
            }

            known_facts_dict[country] = known_facts

        except FileNotFoundError:
            print(f"Warning: Analysis file not found for {country} - {analysis_file}")
            known_facts_dict[country] = None
        except KeyError as e:
            print(f"Warning: Missing expected key {e} in analysis file for {country}")
            known_facts_dict[country] = None

    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[
            {"role": "user",
             "content": f"As a pharmacy product analyst to help the pharmacy store, please analyze the sales for the drug {category} in Germany and Poland, within 500 words respectively."},
            {"role": "assistant",
             "content": "Of course, I can help you with that. Could you provide me with some sales information about the product in Germany and Poland?"},
            {"role": "user",
             "content": f"Here are the facts for Germany: {known_facts_dict['Germany']}\n\nHere are the facts for Poland: {known_facts_dict['Poland']}"}
        ],
        tools=[
            {
                "type": "retrieval",
                "retrieval": {
                    "knowledge_id": sales_knowledge_id,
                    "prompt_template": "Search from\n\"\"\"\n{{知识}}\n\"\"\"\n to figure out the relevant knowledge of \n\"\"\"\n{{用户输入/对话}}\n\"\"\"\n，you shall form your response with respect to the knowledge you found, if you cannot find out any knowledge from the document(or the document is empty), you shall answer the question by your own proffesional ability. Please cite if you use any knowledge from document, \n Do not repeat question，Answer the question directly in English."
                }
            }
            ],
    )
    sales_analysis = response.choices[0].message.content

    messages.append((commodity_name,
                     f"The drug '{commodity_name}' belongs to category '{category}'.\n\nAnalysis:\n{sales_analysis}"))
    return messages, known_facts_dict


# Function to send a message to the chatbot to keep talking (remember using the history)
def send_message(chat_history, message):
    messages = []
    print(chat_history)
    for i in range(0, len(chat_history)):
        messages.append({
            "role": "user",
            "content": chat_history[i][0]
        })
        messages.append({
            "role": "assistant",
            "content": chat_history[i][1]
        })
    messages.append({
        "role": "user",
        "content": message
    })
    # using history to keep the conversation
    print(messages)
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=messages,
        tools=[
            {
                "type": "retrieval",
                "retrieval": {
                    "knowledge_id": sales_knowledge_id,
                    "prompt_template": "从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找问题\n\"\"\"\n{{question}}\n\"\"\"\n的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n不要复述问题，使用英语直接开始回答。"
                }
            }
        ],
    )
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    chat_history.append([message,response.choices[0].message.content])
    return chat_history



# Create Gradio Blocks interface
with gr.Blocks() as demo:
    title = "# Sales Assistant Chatbot"
    gr.Markdown(title)
    category = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            commodity_input = gr.Textbox(label="Input Drug Name")
            classify_button = gr.Button("Step1.Classify")
            category_output = gr.Textbox(label="Category", interactive=False)
            suggestions_button = gr.Button("Step2.Generate Analysis")
            # known_facts_output = gr.Textbox(label="Known Facts", interactive=False)
            with gr.Accordion("Analysis Results by Country", open=False):
                germany_facts = gr.JSON(label="Germany Facts", visible=True)
                poland_facts = gr.JSON(label="Poland Facts", visible=True)
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="Pharmacy Store Chatbot", height=600)
            with gr.Row():
                chatbot_input = gr.Textbox(label="Your Message", scale=4)
                send_button = gr.Button("Continue Chat", scale=1)
                # html_button = gr.Button("Open US State Profit Map")
            # with gr.Row():
            #     # 2 images in a row
            #     image1 = gr.Image(value=None, label="Top 10 States Distribution")
            #     image2 = gr.Image(value=None, label="Top 10 Cities Distribution")
            # with gr.Row():
            #     image3 = gr.Image(value=None, label="Seasonal Analysis")
            # with gr.Row():
            #     image4 = gr.HTML(label="Discount VS Sales and Profit")
            #     image5 = gr.HTML(label="Sales Over Time")
            # with gr.Row():
            #     image6 = gr.HTML(label="Ship Mode Distribution")
            #     image7 = gr.HTML(label="Unit Price Profit Discount By Segment")

    classify_button.click(fn=classify_commodity, inputs=commodity_input, outputs=category_output)


    # suggestions_button.click(fn=generate_suggestions, inputs=[chatbot, commodity_input, category_output], outputs=[chatbot, known_facts_output])
    def process_suggestions(chatbot, commodity_name, category):
        updated_chatbot, facts_dict = generate_suggestions(chatbot, commodity_name, category)

        return (
            updated_chatbot,
            facts_dict.get('Germany', {}),
            facts_dict.get('Poland', {})
        )


    suggestions_button.click(
        fn=process_suggestions,
        inputs=[chatbot, commodity_input, category_output],
        outputs=[chatbot, germany_facts, poland_facts]
    )
    send_button.click(fn=send_message, inputs=[chatbot, chatbot_input], outputs=chatbot)
    # html_button.click(fn=open_html, inputs=category_output)

demo.launch(share=True)