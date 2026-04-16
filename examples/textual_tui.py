"""Launch the Xerxes Textual terminal UI."""

from xerxes_agent import Agent, OpenAILLM, RuntimeFeaturesConfig, Xerxes


def main() -> None:
    llm = OpenAILLM(model="gpt-4o-mini")
    Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(
        id="assistant",
        model="gpt-4o-mini",
        instructions="You are a concise coding assistant.",
        functions=[],
    )
    xerxes.register_agent(agent)
    xerxes.create_tui(agent).launch()


if __name__ == "__main__":
    main()
