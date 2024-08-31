import gymnasium as gym
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

from llm.llm_api_agent.abstract_puzzles_agent import AbstractPuzzlesAgent


class ChatGPTPuzzlesAgent(AbstractPuzzlesAgent):
    def __init__(self, *args, api_key=None, model="gpt-4o", **kwargs):
        """
        Initializes the ChatGPTPuzzlesAgent.

        Args:
            *args: Positional arguments passed to the parent class.
            api_key: The API key for accessing the OpenAI API.
            model: The name of the OpenAI model to use (e.g., "gpt-4o").
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, max_retries=5)
        self.model = model

    def format_image_observation(self, observation: gym.core.ObsType) -> str:
        """
        Formats the image observation (NumPy array) into a base64-encoded string for ChatGPT.

        Args:
            observation: The NumPy array representing the image observation.

        Returns:
            The formatted image observation as a base64-encoded string.
        """
        observation = observation["pixels"]
        if observation.shape[0] == 1:
            observation = observation[0]
        img = Image.fromarray(observation, mode="RGB")

        # Use BytesIO to store the image in memory
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    def get_api_response(self, text_prompt: str, formatted_image: any) -> str:
        """
        Gets the response from the OpenAI ChatGPT API.

        Args:
            text_prompt: The text prompt for the language model.
            formatted_image: The formatted image observation (base64-encoded string).

        Returns:
            The language model's response as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{formatted_image}",
                            },
                        },
                    ],
                },
            ],
        )
        response_text = response.choices[0].message.content
        if self.print_response:
            print(response_text)
        return response_text
