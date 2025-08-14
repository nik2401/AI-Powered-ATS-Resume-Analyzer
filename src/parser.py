import pymupdf # imports the pymupdf library


def pdf_parser(filename):
  text = []
  doc = pymupdf.open(filename) # open a document
  for page in doc: # iterate the document pages
    text = page.get_text() # get plain text encoded as UTF-8
  doc.close() # close the document
  return text

def job_desc():
    print("Paste the job description. Type END on a new line to finish:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)