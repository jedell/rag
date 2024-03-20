import glob
import json

import asyncio
import glob
import json

async def process_user_questions():
    processed_data = []
    files = glob.glob('data/generated/raw/*.json')
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                user_text = item['user']
                print("Before", user_text)
                # if the text contains the words "passage" or "excerpt" or "text" or "scene" do the processing 
                split_text = user_text.split(" ")
                if "passage" in split_text or "excerpt" in split_text or "text" in split_text or "scene" in split_text or "snippet" in split_text:
                    # Perform the async operation on user_text
                    processed_text = await some_async_operation(user_text)
                else:
                    # Perform the async operation on user_text
                    processed_text = 'No rephrasing needed'
                print(f"Processed: {processed_text}")
                # Create a new object with the processed text
                if processed_text == 'No rephrasing needed':
                    new_item = {**item, 'user': user_text}
                else:
                    new_item = {**item, 'user': processed_text}
                processed_data.append(new_item)
    # Save all processed data to one big json file
    with open('data/generated/processed/all_processed_data.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)


from openai import AsyncOpenAI

client = AsyncOpenAI()

# [[[User]]] I hope this suffices for you!

async def some_async_operation(text):
    prompt = f"""Please remove phrases related to a reference of texts, excerpts, or passages from the question in favor of a general Dune context: '{text}'.
    That is, remove the phrases that ground the question in any specific passage, excerpt or text. Do not remove references to the lore or universe of Dune.
    For example, rephrase questions such as 'What is the meaning of "yodel" in the passage?' to 'What is the meaning of "yodel"?'.
    If the question does not need to be rephrased, please respond with 'No rephrasing needed'.
    Only output the rephrased question and format your response as follows:
    [[[User]]] <response>
    """
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "Rephrase questions to be general and not text-specific."},
            {"role": "user", "content": prompt},
        ]
    )
    processed_text = response.choices[0].message.content
    return processed_text

# asyncio.run(process_user_questions())

import random

async def clean_up_processed_data():
    file_path = 'data/generated/processed/all_processed_data.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Define the phrase to be removed
    phrase_to_remove = " in the Dune universe"
    
    # Iterate through each item in the data list
    for item in data:
        # Randomly decide whether to remove the phrase or not, remove 25% of the time
        if random.choice([True, False]):
            # Check and remove the phrase from 'user' key if present
            if phrase_to_remove in item['user']:
                item['user'] = item['user'].replace(phrase_to_remove, '')
            # Check and remove the phrase from 'assistant' key if present
            if phrase_to_remove in item['assistant']:
                item['assistant'] = item['assistant'].replace(phrase_to_remove, '')
    
    # Write the cleaned data back to the file
    file_path = 'data/generated/processed/all_processed_data_cleaned.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Call the function to start the cleaning process
# asyncio.run(clean_up_processed_data())

# sample random items from all_processed_data_cleaned.json and print them
# file_path = 'data/generated/processed/all_processed_data_cleaned.json'
# with open(file_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# print(random.choice(data))

# iterate through all_processed_data_cleaned.json, keys user, assistant, content
# if '[[[Content]]]' or '[[[User]]]' or '[[[Assistant]]]' in any of the keys remove
file_path = 'data/generated/processed/all_processed_data_cleaned.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if '[[[Content]]]' in item['content']:
        item['content'] = item['content'].replace('[[[Content]]]', '')
    if '[[[User]]]' in item['user']:
        item['user'] = item['user'].replace('[[[User]]]', '')
    if '[[[Assistant]]]' in item['assistant']:
        item['assistant'] = item['assistant'].replace('[[[Assistant]]]', '')

    if '[[[Assistant]]]' in item['user']:
        # remove assistant and everything after
        item['user'] = item['user'].split('[[[Assistant]]]')[0]

    if '[[[Content]]]' in item['assistant']:
        # remove user and everything after
        item['assistant'] = item['assistant'].split('[[[Content]]]')[0]

    if '[[[User]]]' in item['assistant']:
        # remove user and everything after
        item['assistant'] = item['assistant'].split('[[[User]]]')[0]

target_folder = 'data/generated/processed/all_processed_data_cleaned_no_markers_split.json'
with open(target_folder, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

