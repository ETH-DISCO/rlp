import time
import numpy as np
from PIL import Image
import google.generativeai as genai

from llm.llm_api_agent.abstract_puzzles_agent import AbstractPuzzlesAgent


class GeminiPuzzlesAgent(AbstractPuzzlesAgent):
    def __init__(self, *args, api_key=None, model="gemini-1.5-flash", **kwargs):
        """
        Initializes the GeminiPuzzlesAgent.

        Args:
            *args: Positional arguments passed to the parent class.
            api_key: The API key for accessing the Gemini model.
            model: The name of the Gemini model to use.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.api_key = api_key
        self.chat_history = []

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def format_image_observation(self, observation: np.ndarray) -> Image:
        """
        Formats the image observation (NumPy array) into a PIL Image for Gemini.

        Args:
            observation: The NumPy array representing the image observation.

        Returns:
            The formatted image observation as a PIL Image.
        """
        # Assuming observation is a NumPy array representing an image
        observation = observation["pixels"]
        if observation.shape[0] == 1:
            observation = observation[0]
        img = Image.fromarray(observation, mode="RGB")

        return img

    def get_api_response(self, text_prompt: str, formatted_image: Image) -> str:
        """
        Gets the response from the Gemini API, with potential retries.

        Args:
            text_prompt: The text prompt for the language model.
            formatted_image: The formatted image observation.

        Returns:
            The language model's response as a string.
        """
        response = self.get_api_response_with_backoff(text_prompt, formatted_image)
        if self.print_response:
            print(response)
        return response

    def get_api_response_with_backoff(
        self, text_prompt: str, formatted_image: Image, max_retries=6, initial_delay=2
    ):
        """
        Gets an API response with exponential backoff in case of failures.

        Args:
            text_prompt: The text prompt to send to the API.
            formatted_image: The formatted image to send to the API.
            max_retries: The maximum number of retries before giving up.
            initial_delay: The initial delay in seconds before the first retry.

        Returns:
            The API response text if successful, or None if all retries fail.
        """

        retries = 0
        delay = initial_delay

        while retries <= max_retries:
            try:
                response = self.model.generate_content([text_prompt, formatted_image])
                response_text = response.text
                return response_text
            except Exception as e:  # Catch any request exception
                if retries == max_retries:
                    print(f"Max retries reached. Giving up. Error: {e}")
                    return None
                else:
                    print(f"Request failed. Retrying in {delay} seconds. Error: {e}")
                    time.sleep(delay)
                    delay *= 2  # Double the delay for the next retry
                    retries += 1
