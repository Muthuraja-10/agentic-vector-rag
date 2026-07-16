import re


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Replace tabs with a space
        text = text.replace("\t", " ")

        # Remove multiple spaces
        text = re.sub(r" +", " ", text)

        # Remove multiple blank lines
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()