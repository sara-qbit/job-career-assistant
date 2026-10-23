# 11: job_market_stats + charts

**What to build:** The candidate can ask market questions and sees exact numbers as charts (v0.7, part 1).

**Blocked by:** 07

**Status:** ready-for-agent

- [ ] `job_market_stats` offers a fixed set of parameterised SQL statistics (counts by Target Role, work type, remote, location; salary ranges) over all postings or tagged postings; the LLM picks the statistic and parameters via the tool schema and never writes SQL
- [ ] Answers state the population used ("all 123,849 postings" vs "Target-Role postings")
- [ ] Results rendered as charts on MAIN
- [ ] Seam-2 tests: exact counts on fixture data, population label, invalid parameters rejected
