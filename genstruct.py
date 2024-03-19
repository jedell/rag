from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_NAME = 'NousResearch/Genstruct-7B'

# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='cuda', load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_length=8192)

msg =[{
    'title': 'Dune Lore',
    'content': """"Does this mean Duncan was successful?" she asked. "Will the Fremen be our allies?"
"There's nothing definite," he said. "They wish to observe us for a while, Duncan believes. They did, however, promise to stop raiding our outlying villages during a truce period. That's a more important gain than it might seem. Hawat tells me the Fremen were a deep thorn in the Harkonnen side, that the extent of their ravages was a carefully guarded secret. It wouldn't have helped for the Emperor to learn the ineffectiveness of the Harkonnen military." 
"A Fremen housekeeper," Jessica mused, returning to the subject of the Shadout Mapes. "She'll have the all-blue eyes." 
"Don't let the appearance of these people deceive you," he said. "There's a deep strength and healthy vitality in them. I think they'll be everything we need." 
"It's a dangerous gamble," she said. 
"Let's not go into that again," he said.
She forced a smile. "We are committed, no doubt of that." She went through the quick regimen of calmness -- the two deep breaths, the ritual thought, then: "When I assign rooms, is there anything special I should reserve for you?"
"You must teach me someday how you do that," he said, "the way you thrust your worries aside and turn to practical matters. It must be a Bene Gesserit thing." 
"It's a female thing," she said.
He smiled. "Well, assignment of rooms: make certain, I have large office space next my sleeping quarters. There'll be more paper work here than on Caladan. A guard room, of course. That should cover it. Don't worry about security of the house. Hawat's men have been over it in depth." 
"I'm sure they have."
He glanced at his wristwatch. "And you might see that all our timepieces are adjusted for Arrakeen local. I've assigned a tech to take care of it. He'll be along presently." He brushed a strand of her hair back from her forehead. "I must return to the landing field now. The second shuttle's due any minute with my staff reserves." 
"Couldn't Hawat meet them, my Lord? You look so tired." 
"The good Thufir is even busier than I am. You know this planet's infested with Harkonnen intrigues. Besides, I must try persuading some of the trained spice hunters against leaving. They have the option, you know, with the change of fief -- and this planetologist the Emperor and the Landsraad installed as Judge of the Change cannot be bought. He's allowing the opt. About eight hundred trained hands expect to go out on the spice shuttle and there's a Guild cargo ship standing by." 
"My Lord . . . " She broke off, hesitating. 
"Yes?"
He will not be persuaded against trying to make this planet secure for us, she thought. And I cannot use my tricks on him. 
"At what time will you be expecting dinner?" she asked.
That's not what she was going to say, he thought. Ah-h-h-h, my Jessica, would that we were somewhere else, anywhere away from this terrible place -- alone, the two of us, without a care.
"I'll eat in the officers' mess at the field," he said. "Don't expect me until very late. And . . .ah, I'll be sending a guardcar for Paul. I want him to attend our strategy conference." 
He cleared his throat as though to say something else, then, without warning, turned and strode out, headed for the entry where she could hear more boxes being deposited. His voice sounded once from there, commanding and disdainful, the way he always spoke to servants when he was in a hurry: "The Lady Jessica's in the Great Hall. Join her there immediately." 
The outer door slammed.
Jessica turned away, faced the painting of Leto's father. It had been done by the famed artist, Albe, during the Old Duke's middle years. He was portrayed in matador costume with a magenta cape flung over his left arm. The face looked young, hardly older than Leto's now, and with the same hawk features, the same gray stare. She clenched her fists at her sides, glared at the painting. 
"Damn you! Damn you! Damn you!" she whispered. 
"What are your orders, Noble Born?" 
It was a woman's voice, thin and stringy.
Jessica whirled, stared down at a knobby, gray-haired woman in a shapeless sack dress of bondsman brown. The woman looked as wrinkled and desiccated as any member of the mob that had greeted them along the way from the landing field that morning. Every native she had seen on this planet, Jessica thought, looked prune dry and undernourished. Yet, Leto had said they were strong and vital. And there were the eyes, of course -- that wash of deepest, darkest blue without any white -- secretive, mysterious. Jessica forced herself not to stare.
The woman gave a stiff-necked nod, said: "I am called the Shadout Mapes, Noble Born. What are your orders?"
"You may refer to me as 'my Lady,' " Jessica said. "I'm not noble born. I'm the bound concubine of the Duke Leto." 
Again that strange nod, and the woman peered upward at Jessica with a sly questioning, "There's a wife, then?"
"There is not, nor has there ever been. I am the Duke's only . . . companion, the mother of his heir-designate."
Even as she spoke, Jessica laughed inwardly at the pride behind her words. What was it St. Augustine said? she asked herself. "The mind commands the body and it obeys. The mind orders itself and meets resistance." Yes -- I am meeting more resistance lately. I could use a quiet retreat by myself.
A weird cry sounded from the road outside the house. It was repeated: "Soo-soo-Sook! Soo-soo-Sook!" Then: "Ikhut-eigh! Ikhut-eigh!" And again: "Soo-soo-Sook!"
"What is that?" Jessica asked. "I heard it several times as we drove through the streets this morning."
"Only a water-seller, my Lady. But you've no need to interest yourself in such as they. The cistern here holds fifty thousand liters and it's always kept full." She glanced down at her dress. "Why, you know, my Lady, I don't even have to wear my stillsuit here?" She cackled. "And me not even dead!"
Jessica hesitated, wanting to question this Fremen woman, needing data to guide her. But bringing order of the confusion in the castle was more imperative. Still, she found the thought unsettling that water was a major mark of wealth here.
"My husband told me of your title, Shadout," Jessica said. "I recognized the word. It's a very ancient word."
"You know the ancient tongues then?" Mapes asked, and she waited with an odd intensity.
"Tongues are the Bene Gesserit's first learning," Jessica said. "I know the Bhotani Jib and the Chakobsa, all the hunting languages." 
Mapes nodded. "Just as the legend says."
And Jessica wondered: Why do I play out this sham? But the Bene Gesserit ways were devious and compelling.
"I know the Dark Things and the ways of the Great Mother," Jessica said. She read the more obvious signs in Mapes' actions and appearance, the petit betrayals. "Miseces prejia," she said in the Chakobsa tongue. "Andral t're pera! Trada cik buscakri miseces perakri --" 
Mapes took a backward step, appeared poised to flee. 
"I know many things." Jessica said. "I know that you have borne children, that you have lost loved ones, that you have hidden in fear and that you have done violence and will yet do more violence. I know many things." 
In a low voice, Mapes said: "I meant no offense, my Lady." 
"You speak of the legend and seek answers," Jessica said. "Beware the answers you may find. I know you came prepared for violence with a weapon in your bodice." 
"My Lady, I . . . "
"There's a remote possibility you could draw my life's blood," Jessica said, "but in so doing you'd bring down more ruin than your wildest fears could imagine. There are worse things than dying, you know -- even for an entire people."
"My Lady!" Mapes pleaded. She appeared about to fall to her knees. "The weapon was sent as a gift to you should you prove to be the One."
"And as the means of my death should I prove otherwise," Jessica said. She waited in the seeming relaxation that made the Bene Gesserit-trained so terrifying in combat.

The following is a question answering chat interation between a user and an AI assistant based on the above text and related to the Dune universe.
The user is a fan of the Dune series and is asking the AI assistant a question about the lore of Dune in order to gain more understanding of the Dune universe.
The interaction can consist of multiple questions and answers with the questions starting as general and becoming more specific as the interaction progresses.
Questions are related to the lore of Dune and can be answered from the above text. The user has no knowledge of the above text and does not reference it in their question.
The AI assistant's response to the question is grounded in information in the text.
"""
}]

# inputs = tokenizer.apply_chat_template(msg, return_tensors='pt').cuda()

# print(tokenizer.decode(model.generate(inputs, max_new_tokens=2048)[0]).split(tokenizer.eos_token)[0])

import os
import asyncio
import json
import time
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def ask_dune_question(content):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": content},
        ]
    )
    return response.choices[0].message.content

async def generate_dune_data():
    source_folders = [f"data/chunks/dune{i}" for i in range(1, 7)]
    char_count = 0
    for i, source_folder in enumerate(source_folders, start=1):
        print(f"Generating data for dune{i}")
        tasks = []
        responses = []
        target_folder = f"data/generated/qa/dune{i}"
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        all_data = []
        fail_data = []
        start = time.time()
        for filename in os.listdir(source_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(source_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if char_count / 4 > 70000:
                        # await current tasks and append to responses
                        responses.extend(await asyncio.gather(*tasks))
                        tasks = []
                        char_count = 0
                        # sleep for what is remaining in 1 minutes from start
                        time_elapsed = time.time() - start
                        remaining = 60 - time_elapsed
                        print(f"Sleeping for {remaining} seconds to avoid token per minute limit")
                        if remaining > 0:
                            time.sleep(remaining)
                        else:
                            time.sleep(1)
                        start = time.time()
                        char_count = 0
                    char_count += len(content)
                    system_message = {
                        'system': """Generate a question answering chat interaction between a user and an AI assistant based on the text snippet from one of the Dune novels by Frank Herbert below (marked [[[Content]]]).
The generated chat question can be answered from the text in relation to the lore of Dune. The user has no knowledge of the below text and should not reference it in the question (i.e. using phrases such as "in the passaged" or "in the text").
Again, the generated question should make no reference to any excerpts of the text and should not be a direct question about the text, but rather a question about the lore of Dune that can be answered from the content and analysis of the text.
The AI assistant's response to the question should be grounded in facts, information, or analysis of the text content.
If required, generate a multi-turn interaction where questions go from general to more specific.
Follow this format for questions and answers:
[[[User]]] <question>
[[[Assistant]]] <response>
""",
                        'content': f"[[[Content]]] {content}"
                    }
                    system = system_message['system'] + system_message['content']
                    tasks.append(ask_dune_question(system))
                    # response = await ask_dune_question(system)
                    # tasks.append(response)
        # responses = await asyncio.gather(*tasks)
        responses.extend(await asyncio.gather(*tasks))
        # responses = tasks
        for response, filename in zip(responses, os.listdir(source_folder)):
            # read content
            file_path = os.path.join(source_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            try:
                user_splits = response.split('[[[User]]]')
                assistant_splits = response.split('[[[Assistant]]]')
                if len(user_splits) > 2 and len(assistant_splits) > 2:  # Multiple turns detected
                    for i in range(1, len(user_splits)):
                        try:
                            question = user_splits[i].split('[[[Assistant]]]')[0].strip()
                            answer = assistant_splits[i].strip()
                            data = {
                                'file': filename,
                                'content': content,
                                'user': question,
                                'assistant': answer
                            }
                            all_data.append(data)
                        except IndexError:  # In case of mismatch in counts, which should not happen
                            continue
                else:  # Single turn
                    question = user_splits[1].split('[[[Assistant]]]')[0].strip()
                    answer = assistant_splits[1].split('[[[User]]]')[0].strip()
                    data = {
                        'file': filename,
                        'content': content,
                        'user': question,
                        'assistant': answer
                    }
                    all_data.append(data)
            except Exception as e:
                fail_data.append({
                    'file': filename,
                    'error': str(e),
                    'response': response
                })
        output_file = os.path.join(target_folder, f"dune{i}_qa7.json")
        with open(output_file, 'w', encoding='utf-8') as out_file:
            json.dump(all_data, out_file, ensure_ascii=False, indent=4)
        if fail_data:
            fail_file = os.path.join(target_folder, f"dune{i}_fail.json")
            with open(fail_file, 'w', encoding='utf-8') as out_file:
                json.dump(fail_data, out_file, ensure_ascii=False, indent=4)

asyncio.run(generate_dune_data())

