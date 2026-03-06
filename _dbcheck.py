import sys
sys.path.insert(0, '/Users/s748779/CEP_AI/anzsic_mapping_v1')
from dotenv import load_dotenv
load_dotenv('/Users/s748779/CEP_AI/anzsic_mapping_v1/.env', override=True)
from prod.config.settings import Settings
from prod.adapters.postgres_db import PostgresDatabaseAdapter

s = Settings()
db = PostgresDatabaseAdapter(s)

queries = [
    'fixes pipes in industries for AC',
    'plumber pipe repair',
    'mobile mechanic',
    'registered nurse',
    'software developer',
]

for q in queries:
    hits = db.fts_search(q, limit=5)
    print(f'[{len(hits)} hits] {q}')
    for code, rank in hits[:3]:
        recs = db.fetch_by_codes([code])
        desc = recs[code]['anzsic_desc'] if code in recs else '?'
        print(f'   {rank}. {code} | {desc}')

db.close()

