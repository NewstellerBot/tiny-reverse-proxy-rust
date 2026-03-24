# Security Policy

## Reporting

Do not file public GitHub issues for suspected vulnerabilities.

Send security reports privately to the maintainers through the repository security advisory flow or
the contact address documented in the repository settings.

## In scope

- authentication and authorization flaws
- key leakage or privilege escalation
- unsafe management API exposure
- request smuggling, header confusion, or cache poisoning issues
- dependency issues that expose real exploit paths in this project

## Out of scope

- unsupported deployment topologies
- local development setups without secrets protection
- behavior explicitly marked as preview or experimental unless it creates a broader exploit path
- self-DoS caused by clearly unsafe operator configuration

## Disclosure expectations

- give maintainers reasonable time to confirm and patch
- coordinated disclosure is preferred
- fixes should default to the safer behavior, with temporary escape hatches only when necessary
