from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from pydantic import BaseModel
from typing import List

import datetime, json, random, string, os

class BigQueryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class RO_Service(BaseModel):
    ro_service_num: int
    ro_service_desc: str
    ro_service_amt: Decimal

class RO(BaseModel):
    country: str
    dealer: str
    ro_num: int
    vin_num: str
    ro_open_date: date
    ro_close_date: date
    total_ro_amt: Decimal
    ro_service_num: int
    ro_service_desc: str
    ro_service_amt: Decimal

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Learning/AI/Key/gen-lang-client-0398817262-d752d424f736.json"

project_id="gen-lang-client-0398817262"
dataset_id="GenAi"
table_id="vin_sh"
full_table_id=f"{project_id}.{dataset_id}.{table_id}"

def generate_rand_str(len: int, letters) -> str:
    return ''.join(random.choice(letters) for i in range(len))

def generate_rand_date(start_date_str="2020-01-01") -> date:
    start_date=date.fromisoformat(start_date_str)
    end_date=date.today()

    days_between = (end_date - start_date).days
    rand_days=random.randrange(days_between)
    return (start_date + timedelta(days=rand_days))


def generate_ros() -> List[RO]:
    ro_list : List[RO] = []
    for char in string.ascii_uppercase:
        vin_num=char*17
        country=generate_rand_str(3, string.ascii_uppercase)
        dealer=generate_rand_str(5, string.ascii_uppercase + string.digits)
        ro_num=generate_rand_str(5, string.digits)
        ro_open_date=generate_rand_date()
        ro_close_date=generate_rand_date(ro_open_date.strftime('%Y-%m-%d'))
        
        total_ro_amt=Decimal('0.00')
        service_list : List[RO_Service] = []
        for i in range(random.randint(1,5)):
            ro_service_num=i+1
            ro_service_desc=generate_rand_str(10, string.ascii_letters)
            ro_service_amt=Decimal(str(random.uniform(1.0, 50.0))).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP)
            total_ro_amt+=ro_service_amt
            service_list.append(RO_Service(ro_service_num=ro_service_num, 
                    ro_service_desc=ro_service_desc, ro_service_amt=ro_service_amt))
            
        for service in service_list:
           ro_list.append(RO(country=country, dealer=dealer, ro_num=ro_num, vin_num=vin_num,
                ro_open_date=ro_open_date, ro_close_date=ro_close_date, total_ro_amt=total_ro_amt, 
                ro_service_num=service.ro_service_num, ro_service_desc=service.ro_service_desc, 
                ro_service_amt=service.ro_service_amt)) 

    return ro_list


def insert_bq(ro_list : List[RO]):
    bq_client=bigquery.Client(project=project_id)
    schema = [
        bigquery.SchemaField("country", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dealer", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ro_num", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("vin_num", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ro_open_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ro_close_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("total_ro_amt", "NUMERIC", mode="REQUIRED"),
        bigquery.SchemaField("ro_service_num", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("ro_service_desc", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ro_service_amt", "NUMERIC", mode="REQUIRED")
    ]
    table = bigquery.Table(full_table_id, schema=schema)
    try:
        bq_client.get_table(full_table_id)
        print(f"Table -> {full_table_id} exists")
    except NotFound:
        print(f"Table -> {full_table_id} doesn't exist. Creating...")
        bq_client.create_table(table=table)
        print(f"Table -> {full_table_id} created")

    rows_to_insert=[]
    for ro in ro_list:
        rows_to_insert.append(ro.model_dump(mode='json'))

    print(len(rows_to_insert))
    errors=bq_client.insert_rows_json(full_table_id, rows_to_insert)
    if errors:
        print(f"Error inserting to BQ -> {errors}")


def bq_vin_sh(vin : str):
    bq_client=bigquery.Client(project=project_id)
    query=f"""
        SELECT country, dealer, ro_num, vin_num, ro_open_date, ro_close_date, 
        total_ro_amt, ro_service_num, ro_service_desc, ro_service_amt
        FROM `{full_table_id}`
        WHERE vin_num = '{vin}'
    """
    query_job=bq_client.query(query)
    results=query_job.result()
    rows_as_dicts = [dict(row) for row in results]
    return json.dumps(rows_as_dicts, cls=BigQueryEncoder)


def bq_vin_service_count(vin : str):
    bq_client=bigquery.Client(project=project_id)
    query=f"""
        SELECT vin_num, ro_num, count(*) as `ro_count`
        FROM `{full_table_id}`
        WHERE vin_num = '{vin}'
        GROUP BY vin_num, ro_num
    """
    query_job=bq_client.query(query)
    results=query_job.result()
    rows_as_dicts = [dict(row) for row in results]
    return json.dumps(rows_as_dicts, cls=BigQueryEncoder)

def main():
#    ro_list = generate_ros()
#    insert_bq(ro_list)
    op=bq_vin_sh("ZZZZZZZZZZZZZZZZZ")
    print(op)

    op=bq_vin_service_count("ZZZZZZZZZZZZZZZZZ")
    print(op)

if __name__ == "__main__":
    main()
