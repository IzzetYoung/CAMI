import copy
import os
import regex


def heuristic_moderator(context):
    if "goodbye" in context[-1].lower() or "good bye" in context[-1].lower():
        return True
    words1 = set(context[-1].lower().split())
    words2 = set(context[-3].lower().split())
    intersection = words1.intersection(words2)
    overlap_degree = len(intersection) / min(len(words1), len(words2))
    if overlap_degree > 0.9:
        return True
    return False


class Env:
    def __init__(
        self,
        client,
        counselor,
        max_turns=20,
        initial_context=[
            "Counselor: Hello. How are you?",
            "Client: I am good. What about you?",
        ],
        output_file=None,
    ):
        self.client = client
        self.counselor = counselor
        self.conversation = copy.deepcopy(initial_context)
        self.max_turns = max_turns
        if output_file:
            directory = os.path.dirname(output_file)
            os.makedirs(directory, exist_ok=True)
            self.output_file = open(output_file, "w")
            for context in self.conversation:
                self.output_file.write(context + "\n")
        else:
            self.output_file = None
            for context in self.conversation:
                print(context)

    def output(self, utterance):
        if self.output_file:
            self.output_file.write(utterance + "\n")
        else:
            print(utterance)

    def clean_utterance(self, utterance):
        utterance = regex.sub(r"\[(?:[^\[\]]++|(?R))*\]", "", utterance)
        return utterance

    def interact(self):
        for _ in range(self.max_turns):
            counselor_response = self.counselor.reply()
            counselor_response = counselor_response.replace("\n", " ")
            self.output(counselor_response)
            counselor_response = self.clean_utterance(counselor_response)
            self.conversation.append(counselor_response)
            self.client.receive(counselor_response)
            if heuristic_moderator(self.conversation):
                break
            client_response = self.client.reply()
            client_response = client_response.replace("\n", " ")
            self.output(client_response)
            if (
                "You are motivated because" in client_response
                or "You should highlight current state and engagement, express a desire to end the current session"
                in client_response
            ):
                break
            client_response = self.clean_utterance(client_response)
            self.conversation.append(client_response)
            self.counselor.receive(client_response)
            if heuristic_moderator(self.conversation):
                break
