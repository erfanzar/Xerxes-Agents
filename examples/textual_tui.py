"""Launch the Calute Textual terminal UI."""

from calute import Agent, Calute, OpenAILLM, RuntimeFeaturesConfig


def main() -> None:
    llm = OpenAILLM(model="gpt-4o-mini")
    calute = Calute(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(
        id="assistant",
        model="gpt-4o-mini",
        instructions="You are a concise coding assistant.",
        functions=[],
    )
    calute.register_agent(agent)
    calute.create_tui(agent).launch()


if __name__ == "__main__":
    main()
