import asyncio
from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv

load_dotenv()


async def main():
    llm = ChatGoogle(model="gemini-3.1-flash-lite-preview")
    agent = Agent(
        task=(
            "Go to https://erp.smartmes.com/platform/Frames/Login.aspx?ReturnUrl=%2fplatform, "
            "click the `Active Directory` button, "
            "wait for the Microsoft login page redirect, "
            "use this username and password to login: `chew.cheewai@ctiresources.com.my` / `User@1234!`, "
            "when asked about staying signed in, click `No`, "
            "in the Acumatica main page after login, go to Favourites "
            "click Expenses Claims, "
            "tell me what is the total amount of my claims."
        ),
        llm=llm,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
