from llama_index.core import Document
import csv
from typing import List

def csv_load(file) -> List[Document]:
    
    with open(file) as fp:
        csv_reader = csv.reader(fp)
        next(csv_reader)
        documents = []
        for row in csv_reader:        
            document = Document(
                text=row[4],
                metadata={
                    "title": row[1],
                    "link": row[2],
                    "date": row[3],
                    "tags": row[5]
                },
                excluded_llm_metadata_keys=["file_name"],
                metadata_seperator="\n",
                metadata_template="{key}: {value}",
                text_template="Metadata: {metadata_str}\n----------\n\nContent: {content}",
            )
            documents.append(document)
    return documents
