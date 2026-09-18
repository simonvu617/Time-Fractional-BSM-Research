# Time-Fractional Black-Scholes Research

This repository is an empirical-research foundation for studying subdiffusive time-fractional Black-Scholes (TFBSM) across stock and ETF option markets and observation frequencies.

## Research question

**How do TFBSM's pricing accuracy and estimated fractional parameter alpha vary across stock and ETF option markets, observation frequencies, and market conditions, compared with classical Black-Scholes?**

The study extends the daily S&P 500 option analysis of [An et al. (2024)](https://doi.org/10.1007/s11075-023-01563-4) to ask whether its findings generalize across underlying assets and persist within the trading day. More observations provide the means to answer those questions; sample size alone is not the intended contribution.

Out-of-sample testing supports the pricing comparison by evaluating observations excluded from parameter estimation. Forecasting later prices and evaluating trading returns are possible extensions, rather than the primary research question. Repricing with an observed future underlying price must be distinguished from a forecast made before that price is known.

The answer is unknown. The project must support a positive, negative, or inconclusive result without changing the test after seeing outcomes.

## Candidate contribution

The strongest proposed focus is to test **how much fitted alpha and any pricing advantage reflect market inactivity, observation frequency, or stale and discretized quotes**. These are candidate experiments, not implemented analyses or established novelty claims:

1. **Compare observation frequencies on matched data.** Construct daily and intraday samples from the same underlying records. Keep contract/date coverage, elapsed calibration windows, model time units, and evaluation targets comparable; report coverage differences. Examine both alpha estimates and pricing errors. Changing frequency need not leave fitted alpha unchanged, and a change alone does not establish a sampling artifact.
2. **Test the interpretation against measured activity.** Examine whether alpha and pricing improvements are associated with trade gaps and observed price persistence after accounting for volatility, spreads, moneyness, and maturity. Report variation across underlying assets and market conditions. Sampled quote persistence, trade arrivals, and underlying inactivity are distinct measurements; their relationship to an option-implied pricing parameter requires testing.
3. **Use controls that can disprove the interpretation.** Apply the same observation and estimation procedure to simulated classical BSM data with rounding, stale observations, and missing records, and to simulated TFBSM data with known parameters. Determine when the procedure falsely reports fractional behavior and when it can recover it. Evaluate held-out prices against comparable non-fractional benchmarks so that extra fitting flexibility is not mistaken for a useful mechanism.

A useful result could establish robust cross-market patterns, identify the observation scales where the model adds information, or show that apparent benefits disappear after measurement controls. Exact frequencies, calibration windows, parameterization, benchmarks, primary metrics, and sample rules remain design choices to settle before evaluating the final comparison.

## Relationship to prior work

The starting literature already includes:

- [An et al. (2024)](https://doi.org/10.1007/s11075-023-01563-4): numerical solution, joint parameter estimation, and daily S&P 500 option-price fitting across market conditions.
- [Orzel and Weron (2010)](https://www.actaphys.uj.edu.pl/R/41/5/1151/pdf): calibration of subdiffusive BSM using constant-price durations and nonconstant observations.
- [Torricelli (2020)](https://doi.org/10.1016/j.physa.2019.123694): trade-duration risk and behavior across time scales in subdiffusive financial models.
- [Dupret and Hainaut (2023)](https://people.math.ethz.ch/~jdupret/Subdiffusive_SVJ.pdf): subdiffusive stochastic volatility and jumps, with an estimation procedure for illiquid asset returns.
- [Shchestyuk and Tyshchenko (2025)](https://www.vmsta.org/journal/VMSTA/article/314/text): an alternative inverse-Gaussian clock and an Airbnb option-pricing illustration.

These papers have different model and estimation assumptions. They establish that empirical calibration, illiquidity, and applications beyond S&P 500 options are existing research topics. The proposed contribution is a controlled empirical investigation of their generality and sensitivity to observation choices. A focused literature check does not establish priority; a broader comparison of prior methods is still needed before claiming novelty.

## Model scope

The candidate model uses an inverse-stable market-time clock with:

```text
0 < alpha <= 1
```

`alpha = 1` must recover classical BSM under identical inputs. This is not a Hurst-exponent, fractional-Brownian-motion, or rough-volatility project.

The mathematical specification, parameter units and time normalization, volatility/alpha identification, and treatment of American exercise and dividends still require explicit choices. A fitted alpha is not automatically a structural estimate of market waiting times.

## Project status

The collector downloads from **ThetaData** with Stocks Pro and Options Pro. It collects hourly observations for the 21 stock/ETF underlyings in `config.py`. The default **contract-entry window is January 1, 2017 through December 31, 2025**, with a fresh selection each exchange day. Each selected option is enrolled once and followed through expiration, including after the entry window ends. Separate membership tables expose the daily cross-section and the accumulated longitudinal cohort using the same market observations. Only completed dates are requested; a run records contracts whose expiration is still in the future. No full-history run has been completed. Separate dividend/split sources are under review below; no external reference feed is connected yet.

| Collected input | Purpose and interpretation |
| --- | --- |
| Hourly underlying and selected-option bid/ask, sizes, conditions, exchanges | Matched pricing observations and quote-quality research. Preserve raw values, duplicates, and vendor fields. |
| Near-close quotes, five minutes before the actual close | A daily comparison observation: normally 15:55 ET, or 12:55 on a 13:00 close. |
| Hourly underlying and selected-option OHLC, VWAP, volume, trade count | Activity within bars under Theta's SIP aggregation rules. Missing bars stay missing; reported zero activity stays zero. |
| Daily underlying and tracked-option EOD reports | Daily OHLC, volume/count, and available report fields. Retain option rows only during each selected contract's tracked dates. An EOD quote is not the near-close quote. |
| Dated quoted/traded contract lists and daily open interest | Historical candidate membership, including quiet OI-only contracts. OI describes the previous trading session's close. |
| Daily cross-section and cohort membership tables | Today's chosen contracts, first enrollment date/time, discovery evidence, current DTE, and which contracts remain tracked. Weekly entry dates are marked for comparison. |
| SOFR and 11 Treasury tenors within the account's access | Raw reported percent rates for later maturity matching; no invented earlier SOFR observations. |

Individual trades and tick-by-tick quote histories are not downloaded. Hourly activity bars cannot reveal exact trade waiting times. The final regular-session bar can be shorter than an hour. See Theta's [OHLC semantics](https://docs.thetadata.us/operations/option_history_ohlc.html), [EOD reports](https://docs.thetadata.us/operations/option_history_eod.html), and [OI timing](https://docs.thetadata.us/operations/option_history_open_interest.html).

### Selection and following contracts

The default is a **daily cross-section plus hourly longitudinal follow-up**. At 10:30 ET each exchange day, the collector selects today's S/K and DTE grid. New identities join the cohort; previously selected identities keep their original first-entry date and continue through expiration. A contract can appear in several daily cross-sections without creating another enrollment or a second copy of its market observations.

| Monthly table | How to use it |
| --- | --- |
| `cross_sections.parquet` | Contracts selected on each `trade_day`, including previously enrolled contracts that fit today's grid again. `selection_times` and that day's session references identify the underlying price used. |
| `contracts.parquet` | All tracked contract/date pairs, including options outside today's grid. Use for following the same contracts over time. |
| `cohort.parquet` | Surviving identities and their original entry evidence, used to resume the next month. |
| `universes.parquet` | Dated candidate evidence used by the selection rule. |

The cross-section is a selected grid at 10:30, not the entire option surface or an hourly reselection. Its moneyness can drift later that day. Use actual contemporaneous S/K when grouping observations by moneyness; do not treat the whole accumulated cohort as today's fresh grid. Missing reference/discovery inputs produce an `incomplete` cross-section with a null count, while a day outside the enrollment schedule is `not_scheduled`. `observed` means selection inputs were available, not that every selected option has a usable quote; observation coverage is reported separately.

For the weekly robustness comparison, `weekly_entry_day` marks the first exchange session of each week in `cross_sections.parquet`. Combine the monthly tables, take each contract's earliest selection on a marked date, and use its shared observations from that date through expiration. A contract first selected on Wednesday and selected again the following Monday joins the weekly sample on Monday, not Wednesday. Calendar holidays and partial first weeks are handled explicitly; month boundaries and resumes do not restart the week. To collect only the smaller weekly-entry design, use `--enrollment-frequency weekly`. The default `daily` run already contains the membership evidence for both designs.

The stock midpoint chooses listed strikes around target S/K ratios and calendar-day maturities. The maturity targets remain 7, 14, 30, 60, and 120 calendar days, with entry DTE between 7 and 180. The S/K grid is denser near one: 0.95, 0.975, 0.99, 1, 1.01, 1.025, 1.05, plus 0.90 and 1.10 for farther-out comparisons. Nearby targets can select the same listed strike. Quotes, activity, tracked membership, and OI remain daily between enrollment days.

Full candidate-universe snapshots (dated quote/trade lists and bulk OI) are collected on enrollment days with underlying-price access: daily by default, or weekly when requested. These document why contracts could enter the sample, including quiet OI-only contracts. On other tracked dates, OI is retained only for tracked contracts; EOD uses that retention rule every day. Daily manifests distinguish an `observed`, `incomplete`, or `not_requested` universe, with null counts for the latter two. Missing tracked OI reports appear in `coverage.missing_option_oi_count`; an absent report never becomes an invented zero. [Theta's OI documentation](https://docs.thetadata.us/operations/option_history_open_interest.html) describes these as reports of the previous session's closing open interest.

Entry limits apply **once, when a contract joins the cohort**. A contract stays tracked when it moves away from the target moneyness, falls below seven days to expiration, or disappears from a later discovery list. An absent observation produces a coverage gap, not removal from the sample. No option-volume, spread, or OI threshold censors quiet contracts.

The first selection day's full requested observations are retained for retrospective research. `first_selected_date` and `first_selected_time` expose the selection timing; default enrollment at 10:30 was not known at 09:30. [Theta's dated lists](https://docs.thetadata.us/operations/option_list_contracts.html) identify contracts quoted or traded during a date and do not establish exact intraday listing times. The fixed modern ticker universe also does not reconstruct historical market membership. Its sector overlap can support ETF-versus-stock comparisons, but low/medium/high activity groups still need to be established from data rather than ticker labels.

### Access and inputs still missing

| Input | Current account and behavior |
| --- | --- |
| Stock/option history | Pro permits history from June 2012, but ticker/date coverage varies. Successfully refreshed date catalogues identify unlisted underlying quote/activity dates to skip; catalogue failures never justify skipping. A listed date does not prove access or complete observations. |
| Dividends and splits | Unavailable from the current Theta v3 API: live requests returned 404 and the [migration guide](https://docs.thetadata.us/Articles/Getting-Started/v2-migration-guide.html) marks them as coming soon. Recorded as gaps, never zero dividends or no splits. |
| Rates before 2024 | Missing with the account's free rate access. A separately held Interest Rates Value tier can be declared with `--rate-subscription value`. |
| SPX and VIX index levels | Missing without a separate index subscription. Indices Pro begins in 2017; Standard in 2022; Value in 2023. Flags declare existing access and do not purchase it. |
| Exact historical option terms | Verified multipliers, adjusted deliverables, exercise/settlement terms, and last trading timestamps remain unavailable/unverified. Empty fields and explicit gaps prevent assuming every stock option delivers 100 shares. |

Product-context labels accompany cohorts: stock/ETF options are ordinarily American and physically settled; SPX/SPXW are European and cash settled, with AM/PM labels respectively. These labels do **not** certify a particular historical contract's terms. AM settlement can end trading before the expiration date; coverage flags at expiration are not automatically vendor errors. See [Theta's root symbology](https://docs.thetadata.us/Articles/Data-And-Requests/Symbology.html) and [Cboe's SPX specifications](https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications).

SPX/SPXW form an opt-in benchmark using the **SPX index level**, with the option root kept separate from its underlying. Preview with `python collector.py --symbols SPX SPXW --start 2019-01-02 --end 2019-01-04 --plan`. With the current account, index-price access is reported missing and these panels cannot enroll contracts by moneyness. If Indices Pro is later held, adding `--index-subscription pro` enables the relevant price requests. The benchmark uses the same XNYS regular-session observation window, not every extended-hours SPX session. Historical SPXQ/SPXPM roots are outside this benchmark.

[An et al.](https://doi.org/10.1007/s11075-023-01563-4) used daily OptionMetrics SPX/SPXW histories, midpoint prices, activity/OI, index prices, and Treasury rates matched to maturity. Cohort continuity and these inputs matter more for comparison than tick data. Theta's history cannot reproduce their 2008 sample. SPY ETF options also differ from their European cash-settled index options. The present collection does not yet supply every input needed for a fully controlled pricing comparison.

### Batching, storage, and resume

Each option root advances chronologically in monthly checkpoints. Four roots run concurrently without waiting at a global end-of-day barrier; all requests share the **eight-slot Pro limit**. Quotes, prices, activity, EOD, and near-close requests use date ranges where supported. Intraday batches split at month boundaries, early closes, and excluded dates. Quote/trade discovery lists follow the configured enrollment schedule; bulk OI remains one request per tracked day, reusing enrollment-day discovery responses. Selected-option intraday requests specify one expiration with all strikes/both rights, then retain only each enrolled contract's date window. The daily and weekly views share these observations; producing another view does not add another set of downloads. EOD keeps the bulk monthly request across expirations and applies the same retention rule after selection. OI/EOD filtering saves stored rows without multiplying requests; it does not reduce those bulk downloads. Request identities include retained date windows. See [monthly quote limits](https://docs.thetadata.us/operations/option_history_quote.html) and [account-wide concurrency](https://docs.thetadata.us/Articles/Data-And-Requests/Concurrent-Requests.html).

Enrollment dates explicitly absent from the underlying quote catalogue do not trigger broad option-chain downloads. They remain reported gaps. Underlying-data gaps do not stop requests, including OI, for an already enrolled cohort. A failed or unavailable catalogue never supplies evidence for skipping.

Successful responses are packed into shared compressed Parquet files. A SQLite index replaces the old per-request metadata and pointer files. Packing preserves response row groups, order, duplicates, and vendor columns; it publishes new locations before removing individual response files. Failed response bytes remain available, and a failed refresh never displaces the last successful response. `--store-raw-payloads` additionally saves full successful CSV responses, including unselected strikes, at extra storage cost.

The output layout is:

```text
index.sqlite3                       # response receipts, latest successes, session summaries, code hashes
parquet/<dataset>/symbol=.../month=.../  # shared response Parquet files
responses/<attempt-id>.parquet      # individual checkpoints awaiting packing
responses/<attempt-id>.csv          # retained raw/failure payloads
collection/<policy-id>/months/      # monthly manifests and incoming-cohort identity
collection/<policy-id>/tables/      # monthly cross_sections, contracts, universes, cohort checkpoints
collection/<policy-id>/availability.csv # one row per entry/follow-up session; no repeated monthly byte sums
collection/<policy-id>/runs/        # exact run scope, outcome, versions and source hashes
coverage/                          # dated vendor catalogue evidence
references/                        # downloaded references and explicit missing-input ledger
```

Response metadata is readable through `RequestStore.metadata(receipt)`. Its `data.path` and `data.row_groups` locate the response inside a shared Parquet file; `RequestStore.read(receipt)` resolves this automatically. The SQLite `responses.meta` column is plain JSON, and `sessions.manifest` contains each day's coverage and selection references. The `code` table maps a code digest to the full package source hashes.

Schema v9 adds the dated cross-section table and makes enrollment frequency explicit. Existing caches and smoke-test outputs remain untouched and readable; identical compatible raw requests can be reused. Earlier collection manifests cannot satisfy this policy, and weekly-only results cannot supply daily selection evidence. Entry-window dates and enrollment/retention rules are part of the policy identity; changing worker counts does not change that identity. A failed discovery month must be retried before advancing that root's cohort. Resume checks the cross-section artifact along with the cohort, completed requests, and other tables, even when coverage gaps were recorded.

A 60-session underlying lookback remains separate from contract enrollment. Reference rates extend over the possible follow-up tail, subject to access; missing corporate events are reported through the study end plus maximum entry DTE. Neither that buffer nor requested follow-through guarantees continuous observations or chooses a calibration window. Runs with known missing inputs return `2`; request/processing failures return `1`.

No validated pricer, calibration routine, or empirical result is included. These changes remain under review before inclusion on `main`.

The September 12 live check of the earlier v6 daily-enrollment policy used a fresh collector cache for SPY/AAPL on June 2–3, 2025: **61 successful requests, 70,532 saved rows, 24.07 seconds, 33 files, 1.63 MB**. Selected quotes/EOD observations were present; absent option activity bars remained explicit gaps. A recovery check reused saved responses with no downloads, and monthly resume checks passed. This bounded check excluded catalogues, references, and the expiration tail. It does not measure the current daily hybrid design or forecast full-history runtime/storage. Offline fixtures cover daily refresh, repeat selections without repeat enrollment, weekly comparison dates, follow-up after the entry window, retention, and missing inputs. Data and measurement reports stay local.

The September 18 offline replay used those saved SPY/AAPL observations with the current daily and weekly modes. Weekly collection planned **48 requests and stored 735,491 bytes**; daily collection planned **61 requests and stored 918,141 bytes**, about 25% more storage in this two-day example. Daily selection added 126 new identities on the second day across the two roots. The weekly selections extracted from the daily run exactly matched the weekly-only selections. Source observations already covered the requested identities, but the replay reconstructed CSV from saved rows and made no network requests. It does not forecast a full-history run or the expiration tail. The underlying batching, contract-key conversion, and compressed storage optimizations remain shared by both modes.

Using the saved underlying-date catalogues, the 2017–2025 plan has about **92,718 daily versus 19,264 weekly quote/trade listing calls**. The default daily hybrid therefore gives up the earlier 79% reduction in those calls to refresh the cross-section. Daily enrollment can also accumulate more contracts and expiration families; total storage and runtime do not scale by a fixed factor of five. The old weekly estimates are not forecasts for this design. Full-history costs still depend on cohort overlap and Theta throughput. Bulk-versus-individual-strike download speed remains unmeasured; the planner keeps bulk expiration requests. A moving `strike_range` could drop tracked contracts after spot moves. `--output-dir` already supports a local SSD directory outside a synced folder; no storage location was changed or sync overhead measured.

### Supplemental dividend and split sources

Reviewed September 12, 2026. Keeping Theta prices and adding a separately identified corporate-action source preserves a consistent pricing feed. It does not require replacing Theta or inferring events from price jumps. Candidate access is documented, not validated across our 21 tickers:

| Source | Documented access | Remaining check |
| --- | --- | --- |
| [Alpha Vantage corporate actions](https://www.alphavantage.co/documentation/#dividends) | Historical/declared dividends and historical splits; endpoints link to free keys. [Free service](https://www.alphavantage.co/support/) permits 25 requests/day for most datasets. | Test a personal key for all selected stocks/ETFs and the full study window. Public demo calls returned only an API-key notice, so event coverage and historical amount conventions remain unverified. |
| Massive Stocks Starter: [dividends](https://massive.com/docs/rest/stocks/corporate-actions/dividends), [splits](https://massive.com/docs/rest/stocks/corporate-actions/splits) | $29/month; endpoint tables advertise all available history, with dividend records dating to 2000 and split records to 1978. Free Basic is limited to two years. | Confirm per-ticker/ETF coverage and research-use terms with an authenticated sample. Dataset start dates do not guarantee complete records for each ticker. |
| [OCC information memos](https://infomemo.theocc.com/infomemo/search-memo) | Authoritative contract-adjustment notices, including changes to symbols and deliverables. | Link relevant notices to actual held contracts. A stock split ratio alone does not describe every option adjustment. |

A free Alpha Vantage coverage check is a reasonable first step; Massive offers a clearly documented paid historical alternative. No account, purchase, credential, or third-party downloader is added by this review. Theta gaps remain explicit until actual reference records are collected.

Preserve each event's source, identifier, retrieval date, raw amount/ratio, currency, effective/ex-date, and available declaration, record, and payment dates. For example, Massive distinguishes historical `cash_amount` from today's-share-basis `split_adjusted_cash_amount`: keep both as reported rather than mixing adjusted dividends with unadjusted Theta spot prices and option strikes. A dividend eventually paid is not automatically information known at an earlier option observation. Declaration dates help, but do not establish intraday availability or a complete history of vendor revisions. Future undeclared dividends still require an explicit expectation method; realized payouts must not silently supply it.

Corporate-action records support event identification and adjustment handling. They do not themselves provide an American-option exercise model, certify historical deliverables, or supply SPX dividend carry and settlement values. Those research inputs remain separate requirements.

## Running and reading the collector

Use Python 3.11 or newer and install the runtime dependencies:

```sh
python -m pip install -r requirements.txt
python collector.py --symbols SPY --start 2025-01-02 --end 2025-01-03 --plan
```

`collector.py` is the launcher. The same arguments work with `python -m tfbsm_collector`. `--plan` previews the scope without downloading or creating output. Downloads require Theta Terminal v3. The default output remains `data/multi_year_bsm_backtest_output` under the repository root, including when the launcher is called from another directory.

To preview the full default Pro history, run `python collector.py --plan`. Keep the default reference tiers for Stocks Pro and Options Pro alone. If the account later also has Indices Pro and Interest Rates Value, declare them with `--index-subscription pro --rate-subscription value`; the preview shows any remaining access gaps before downloading.

The implementation has ten modules with distinct jobs. Start with `config.py`, then `Collector.run()` in `workflow.py`: it checks date coverage, collects references, and collects stock/option sessions. `Collector.collect_month()` shows dated entry selection followed by batched cohort collection.

One `CollectorConfig` holds the requested symbols, dates, mode, rate series, and collection settings. The CLI parses and previews this configuration; it does not manage collection workers or saved run status. Request builders read the same configuration instead of receiving repeated copies of the run scope. Storage owns output paths and the response-reuse check shared by cache lookup and session resume. Coverage owns both observation checks and the resulting reports.

| File | What to read it for |
| --- | --- |
| [`config.py`](tfbsm_collector/config.py) | Study universe, DTE and S/K targets, sampling times, subscription limits, and download settings. |
| [`workflow.py`](tfbsm_collector/workflow.py) | `Collector.run` and `collect_month`: run order, reference/session collection, cancellation, and final run status. |
| [`planning.py`](tfbsm_collector/planning.py) | Exact Theta request identities, required fields, clock meanings, and calendar-aware request construction. |
| [`selection.py`](tfbsm_collector/selection.py) | Dated contract discovery, stock selection references, and the maturity/moneyness grid. |
| [`transport.py`](tfbsm_collector/transport.py) | HTTP streaming, shared request limits, retry behavior, and cancellation. |
| [`validation.py`](tfbsm_collector/validation.py) | Strict CSV parsing, UTC clock columns, response identity checks, and raw quality diagnostics. |
| [`storage.py`](tfbsm_collector/storage.py) | Output paths, atomic files, response retention, shared resume checks, and output locking. |
| [`coverage.py`](tfbsm_collector/coverage.py) | Date catalogues, missing contracts/sample times, daily report presence, and the final availability CSV. |
| [`provenance.py`](tfbsm_collector/provenance.py) | Source fingerprints, file hashes, collection times, and dependency versions. |
| [`cli.py`](tfbsm_collector/cli.py) | Command-line options, scope preview, and entry into the collection workflow. |

Docstrings and comments follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html): docstrings describe each interface with `Args`, `Returns`/`Yields`, and `Raises` where useful; nearby comments explain financial and collection decisions. Module imports make it clear which file owns an operation. Ruff checks the Google docstring convention and keeps formatting consistent through `pyproject.toml`.

A package-layout change alone does not require a new schema version. This update does, because following contracts changes the sample and resume dependencies. Source fingerprints cover the launcher and every package module. Collection inputs and downloaded market values remain separate from future pricing and analysis code.

Run the maintained offline checks without Theta credentials or a running terminal:

```sh
python -m unittest discover -s tests -p "test_collector.py"
```

For formatting and documentation checks, install Ruff with `python -m pip install ruff`, then run:

```sh
ruff check collector.py tfbsm_collector tests/test_collector.py
ruff format --check collector.py tfbsm_collector tests/test_collector.py
```

These checks cover synthetic collection, selected-row retention, malformed responses, dates across parsing batches, coverage gaps, resume, interruption cleanup, Pro concurrency, and separate reference access. They do not establish live vendor access or historical completeness.

## License, citation, and data

Original code is licensed under the [MIT License](LICENSE), copyright 2026 Simon Vu. Citation metadata are in [CITATION.cff](CITATION.cff).

The MIT License covers this repository's original software. It does not grant permission to redistribute ThetaData responses, third-party papers, or other externally owned material. Generated data and results remain local and are ignored by Git.
