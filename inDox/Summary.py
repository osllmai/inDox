from openai import OpenAI


def summarize(context):
    try:
        client = OpenAI()

        response = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant. Give a detailed summary of the documentation provided"},
                {
                    "role": "user",
                    "content": f"{context}",
                },
            ],
            max_tokens=100,
            model="gpt-3.5-turbo-0125",
        )
        return response.choices[0].message.content

    except Exception as e:
        print(e)
        return e
