Project Name: Research Swarm
Theme: Agent Swarms
Description:

Research Swarm is a multi-agent AI system that answers complex research questions through coordinated collaboration between 5 specialized agents. A Planner decomposes the user's query into parallel subtasks. Multiple Retriever agents search the web simultaneously using DuckDuckGo. A Validator cross-checks key facts with additional searches and assigns a trust score. A Synthesizer merges all verified findings into a coherent, cited answer. Finally, a Critic scores the answer and triggers a retry loop if quality is below threshold — enabling the swarm to self-correct without human intervention. The system is built on an async message-bus architecture where agents communicate exclusively through an orchestrator, making the swarm modular, extensible, and resilient. A real-time web dashboard visualizes all agent activity live as queries are processed.
