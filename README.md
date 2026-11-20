# solmara-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jarcokpahang5-ai/solmara-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jarcokpahang5-ai/solmara-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty gta6-trophy-rarity-forecast-01 ($250 USDC): a Python 3.13 program (standard library only) that forecasts each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate one week after launch, as five quantiles per trophy, scored by pinball loss on the logit scale. It models earners and players-tracked separately — log earners and log players-tracked fitted against log time since release, with the earners slope shrunk toward its tier and global means — so a refresh's two forces (players progressing vs. new players diluting the rate) are separated instead of blurred into one rate trend. When the recorded page has never moved, a two-component mixture also covers the next refresh: with probability 1 − exp(−horizon / 5h) the page refreshes before the target and each rate is multiplied by 0.25, the launch-night refresh's dilution. Deterministic, no network, runs in well under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=yv60esak&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
