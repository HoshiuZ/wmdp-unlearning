
# Serving artifact status

No online inference service is present or claimed. The repository has offline model reconstruction/evaluation and safety simulation entry points:
- scripts/rmu_adaptation_utils.py::load_delta reconstructs the local RMU delta on the exact base.
- safety/four_groups/generate.py and safety/four_groups/finish.py implement the recorded offline generation/policy simulation.
- safety/qwen3guard/moderator.py implements the standalone classifier adapter.

This is not a deployment-ready service. Missing items include a serving API, authentication/authorization, request logging and retention policy, operational monitoring, service-level latency/load tests, deployment packaging, and a recipient-accessible model artifact store. No service was developed in this delivery phase.
