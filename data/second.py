import asyncio
from typing import List, Dict, Any
import httpx
from tabulate import tabulate


async def fetch_users(client: httpx.AsyncClient, url: str) -> List[Dict[str, Any]]:
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


def prepare_table_data(users: List[Dict[str, Any]]) -> List[List[Any]]:
    return [
        [
            user['id'],
            user['name'],
            user['email'],
            user['address']['city'],
            user['phone']
        ]
        for user in users
    ]


def display_table(table_data: List[List[Any]]) -> None:
    headers = ["ID", "Name", "Email", "City", "Phone"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


async def main():
    url = "https://jsonplaceholder.typicode.com/users"
    
    async with httpx.AsyncClient() as client:
        try:
            users = await fetch_users(client, url)
            table_data = prepare_table_data(users)
            display_table(table_data)
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())