from langchain_community.document_loaders import UnstructuredURLLoader

url = "https://en.wikipedia.org/wiki/Amazon_Redshift"
print(f"Testing URL: {url}")

loader = UnstructuredURLLoader(urls=[url])
data = loader.load()

print(f"Number of documents loaded: {len(data)}")

if len(data) > 0:
    print(f"Content length: {len(data[0].page_content)} characters")
    print(f"Content preview: {data[0].page_content[:300]}")
else:
    print("EMPTY — scraper returned nothing")
    print("This means the site is blocking the scraper")
