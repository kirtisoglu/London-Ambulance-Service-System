import re, pathlib
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()

def rep(old, new, cnt=1):
    global s
    pat = re.escape(old).replace('\\ ', r'\s+'); n = len(re.findall(pat, s)); assert n == cnt, (old[:70], n)
    s = re.sub(pat, lambda m: new, s)

NEW = r"""\subsection{London Ambulance Service Case Study}
\label{sec:exp-emergency}
% =========================================================================
This case study evaluates \textsc{FalCom} on a large-scale hierarchical emergency medical service network based on the London Ambulance Service (LAS), the busiest urban ambulance service in the United Kingdom. LAS serves approximately $9$ million residents through $66$ ambulance stations organized into $21$ operational Groups within $5$ sectors. Unlike previous optimization studies on LAS~\citep{mccormack2015simulation}, which produce a single recommended plan, our goal is to characterize a reference distribution over feasible hierarchical configurations, the stationary law of the implemented chain, and to read the operational layout against it. We run FalCom on the real stations, with no artificial candidates, and use the ensemble in three ways: for the structure that most sampled plans share, for the role of each station, and as a reference against which the operational layout is measured. The last reading yields an upper bound, within a static assignment model and at this calibration, on what the current layout would give up by satisfying workload balance.

\subsubsection{Data}\label{sec:las-data}
The base graph $G^1$ has $|V^1| = 4{,}994$ nodes, the Lower layer Super Output Areas (LSOAs) of Greater London on 2021 boundaries, joined by $|E^1| = 14{,}550$ rook-adjacency edges (Figure~\ref{fig:las-instance}). LSOAs are the small-area unit of NHS planning and of the recent spatial-epidemiology work on LAS~\citep{murphy2024socio}. The level-1 candidate set $F^1$ is the set of $66$ LSOAs that contain an active LAS ambulance station, and nothing else. In the sampler any open station can serve as the level-2 facility of its superdistrict, by the rule given below; the $21$ documented Group headquarters~\citep{las_departments2025}, all of them active stations, enter only the benchmark state. Assumption~\ref{asm:facility-density} fails on this instance by a wide margin. The chain runs on the real stations because the counting predicate of Section~\ref{sec:capacitated} keeps the recursion closable, and the cost of the sparse candidate set is paid in rejections, which we report. Demand $d_v$ is the annual number of attended incidents in LSOA $v$: the 2010 to 2015 mean of the monthly borough totals that LAS publishes~\citep{las_incidents}, about $1.065$ million incidents a year, apportioned to the LSOAs of each borough in proportion to the ONS mid-2024 population estimates~\citep{ons_sape2024}. The total is corroborated by the nearly $1.1$ million incidents attended in 2024/25~\citep{las_ar_2024_25}. Travel times are shortest-path driving times on the OpenStreetMap road network from every station to every LSOA centroid, with free-flow speeds scaled by $1.6$ for driving under lights and sirens; for the four LSOAs that the routing engine cannot reach from any station, one each in Hillingdon, Newham, Bromley and Camden, $0.1\%$ of demand and none hosting a station, straight-line distance at $25$ km/h is used. Variation within a borough is thus a proxy, daytime hot spots are smoothed into their residential surroundings, and the speed factor is an assumption. Appendix~\ref{app:slas} documents the assembly. Absolute energies and drive times inherit these approximations. A uniform speed factor rescales every road drive time and leaves the energies, the $1$-medians, the open frequencies and the ratio of the benchmark's energy to the ensemble's unchanged, although not the eight-minute share; the demand proxy is tested directly in Section~\ref{sec:las-price} by repeating the study under a different within-borough apportionment.

\begin{figure}[!htbp]
\centering
\begin{subfigure}[t]{0.495\linewidth}
  \centering
  \includegraphics[width=\linewidth]{figures/las_map.png}
  \caption{LSOA polygons of the LAS catchment.}
  \label{fig:las-lsoas}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.495\linewidth}
  \centering
  \includegraphics[width=\linewidth]{figures/las_dual_graph.png}
  \caption{Rook-adjacency dual graph $G^1$.}
  \label{fig:las-dual}
\end{subfigure}
\caption{The LAS instance. (\subref{fig:las-lsoas}) The $4{,}994$ LSOAs of the catchment; the $48$ Brentwood LSOAs are excluded. (\subref{fig:las-dual}) The dual graph $G^1$ on which the chain runs, with $14{,}550$ edges. Each node carries its integer annual demand.}
\label{fig:las-instance}
\end{figure}

\subsubsection{Calibration}\label{sec:las-calibration}
The capacity unit is a block of ambulances, and its size follows from the published productivity target and from the feasibility results of Appendix~\ref{app:debt}. The 2024/25 Annual Report~\citep{las_ar_2024_25} states a target of $5.2$ patients per crew per twelve-hour shift, rising to $5.4$. Five patients per shift, two shifts a day and $365$ days give $w_{\mathrm{amb}} = 3{,}650$ calls per ambulance per year. With $\varepsilon^1 = 0.15$, the granularity floor requires the per-team tolerance $\varepsilon^1 w_{\mathrm{unit}}$ to exceed the heaviest LSOA demand, $d_{\max} = 1{,}526$, so a unit must hold at least three ambulances. We take blocks of three and $w_{\mathrm{unit}} = 10{,}887$ calls per year, within $0.6\%$ of $3\,w_{\mathrm{amb}}$. The catchment's demand is $\lceil d / w_{\mathrm{unit}} \rceil = 97$ units, and every sampled state carries exactly $97$, as the coverage rule of Section~\ref{sec:rules} requires. Stations carry one to three units, the safe range of Corollary~\ref{cor:app-invariant}, and Groups carry two to six units with at least $\kappa^2_{\min} = 2$ stations each. Both tolerances are $0.15$. By Proposition~\ref{prop:effective}, a sampled one-unit station then serves between $0.72$ and $1.32$ units, a two-unit station between $1.51$ and $2.56$ and a three-unit station between $2.29$ and $3.79$, and every Group between $0.85$ and $1.15$ times its capacity; the loads realized over the stored plans range from $0.74$ to $1.32$ units per unit of capacity. The number of stations a plan opens is not an input. The calibration admits between $33$ and $66$ open stations, and the chain decides.

The unit is a demand quantum, not a vehicle count. The April 2025 fleet inventory lists $583$ double-crewed ambulances across the $21$ Groups, of which roughly seventy percent are on shift at any time, whereas $97$ blocks of three at the published target would be $291$ vehicles. The difference is the gap between the target and realized productivity, vehicles not crewed around the clock, and the 2010 to 2015 demand level. A calibration that bounds the capacity of each station by the fleet based there is a natural refinement that we leave to future work.

The level-2 facility of a superdistrict is the demand-weighted $1$-median among the stations open inside it, $F^2(s) = \{f^1(D) : D \in \Pc^1\}$, the dynamic rule of Section~\ref{sec:falcom}. The Group leg of the energy is therefore a measure of how compactly the stations of a superdistrict sit around their most central member. It is not a dispatch distance: LAS dispatches from its two Emergency Operations Centres rather than from the Group headquarters, so no vehicle travels from a headquarters to every incident of its Group. The operational comparison of Section~\ref{sec:las-price} rests on the station leg, and the Group leg is reported beside it as a descriptive statistic.

As a readout we use the disaggregated median energy
\[
  E(s) \;=\;
  \underbrace{\sum_{D \in \Pc^1} \sum_{v \in D} d_v \, t\bigl(v, f^1(D)\bigr)}_{E_1(s)}
  \;+\;
  \underbrace{\sum_{S \in \Pc^2} \sum_{v \in S} d_v \, t\bigl(v, f^2(S)\bigr)}_{E_2(s)},
\]
the drive-time form of objective~\eqref{eq:milp-median}, in demand-weighted minutes, with the station leg $E_1$ and the Group leg $E_2$ reported separately. Energy plays no part in the sampling, since every feasible proposal is accepted. It is evaluated on the samples, together with three metrics of the station leg, which describe catchment geometry under a static assignment of every LSOA to its catchment station: the demand-weighted mean drive time from the catchment station, its $90$th percentile, and the share of demand within eight minutes' drive of its catchment station.

\subsubsection{The operational layout as a benchmark}
We reconstruct the current operational layout as a state $s_{\mathrm{LAS}}$ in the same space. The five sectors are unions of boroughs, the five Integrated Care Systems of NHS England, and the 2025 list of departments~\citep{las_departments2025}, which lists the stations of each Group, together with the 2023 operational estate map~\citep{las_estate2023} assign each borough of a sector to one of its three to five Groups, so $89.8\%$ of LSOAs inherit their Group from a borough lookup. The remaining LSOAs lie in boroughs that the estate map splits between two Groups. We assign them by station Voronoi within the borough and correct $1.7\%$ by inspection of the map. Within a Group, every LSOA is served by its nearest station in straight-line distance. Seventeen LSOAs, $0.3\%$ of demand, are left detached from their station's catchment on $G^1$ by this rule, across bends of the Thames and railway land, and we move each to the neighbouring station of the same Group with which it shares the longest boundary. Appendix~\ref{app:slas} documents every step. All $66$ catchments and all $21$ Groups of $s_{\mathrm{LAS}}$ are connected on $G^1$.

The benchmark is not a member of the sampled space, for two reasons that are themselves findings. Three Groups (Deptford, Ilford and St Helier) contain a single station, which the rule $\kappa^2_{\min} = 2$ forbids, and the station loads of $s_{\mathrm{LAS}}$ range from $0.19$ to $4.68$ units, far outside the balance windows; its Group loads, $2.0$ to $6.7$ units, lie inside the level-2 windows, so at level $2$ the layout is load-feasible but not count-feasible. The comparison below is therefore between a contiguous hierarchical layout and a sample of plans that satisfy constraints the layout does not. It is also a comparison of static assignment models. Real dispatch sends the nearest available vehicle, not the home station of a fixed catchment, and LAS balances workload dynamically, so the drive-time figures describe catchment geometry rather than service performance, and the balance the windows impose is a static one.

\subsubsection{Protocol and convergence}\label{sec:las-protocol}
Four chains run for $T = 80{,}000$ steps each with uniform cut selection and unconditional acceptance. The initial plan of each chain is built sector by sector, each sector cut against the global per-unit target with its share of the $97$ units fixed by largest remainder (Lemma~\ref{lem:app-telescoping}), so that every initial state carries exactly $\lceil d/w \rceil$ units; it closes at the first or second attempt in under a second, where a global recursion on this instance rarely closes within its retry budget. The sector-wise start leaves the five sectors as the level-2 partition, far outside the level-2 windows; the first accepted step replaces it, and the initial state is excluded from every range reported below. Scalar traces are recorded at every step and the full plan, the station, superdistrict and capacity of every LSOA, every $40$ steps. The first $16{,}000$ steps are discarded as burn-in, so each chain contributes $1{,}600$ plans and the pooled ensemble holds $6{,}400$. A proposal whose recursion does not close, or that would leave a superdistrict with a single station, is rejected and the chain keeps its state. Each chain completes in $61$ to $62$ minutes on one core, $21.5$ to $22.0$ steps per second.

Acceptance is $67.7\%$ to $68.0\%$, and the rejections are a supergraph phenomenon. In $25.6\%$ to $26.1\%$ of proposals none of the $1{,}000$ spanning trees drawn for the supergraph has a pendant subtree inside its window, a tree failure; in $0.3\%$ to $0.4\%$ the supergraph recursion reaches a residual outside the window of its remaining capacity, a demand failure, about one supergraph failure in a hundred; and in $5.9\%$ to $6.3\%$ the re-cut would leave the selected superdistrict with one station. The base-level recursion fails $10$ to $22$ times per chain, fewer than once in $3{,}000$ proposals. The counting predicate thus removes the failure that the sparse candidate set would otherwise cause, and the residual predicate removes the strandings that capacities and counts can detect; without it the same chains accepted $38\%$ of their proposals. What remains is the tree-structural failure of a greedy extraction over $43$ to $61$ supernodes of capacity one to three (Appendix~\ref{app:supergraph-correctness}).

The four chains agree after burn-in (Figure~\ref{fig:las-traces}). Chain means of the energy lie within $0.4\%$ of each other, $9.66$ to $9.70$ million demand-weighted minutes against a within-chain standard deviation of $0.28$ to $0.30$ million. The largest pairwise Kolmogorov--Smirnov distance between chains, on every twentieth step, is $0.066$ for the energy, $0.037$ for the number of stations, $0.020$ for the number of Groups and $0.070$ for the number of cut edges, and split-$\hat R$ on the per-step series is at most $1.003$. The series are autocorrelated, so the critical values of the two-sample test do not apply; as a reference we use the same statistic between the two halves of one chain, which is $0.028$ to $0.073$ for the energy, $0.021$ to $0.067$ for the number of stations, $0.011$ to $0.039$ for the number of Groups and $0.022$ to $0.083$ for the cut-edge count. On every observable the chains differ from each other no more than each differs from itself. The reference calibrates the noise of the statistic; it cannot detect a region of the state space that all four chains miss, and all four start from the same sector-wise construction, so start-independence is weaker evidence here than on the enumerated instances, where the starts were the first, middle and last feasible states in lexicographic order. Effective sample sizes per chain, on the per-step series, range from $426$ to $526$ for the cut-edge count and $395$ to $474$ for the station-leg energy, the two slowest observables, through $631$ to $795$ for the energy, to $2{,}200$ to $3{,}400$ for the number of Groups. Summed over the chains, the $6{,}400$ stored plans are worth about $2{,}970$ independent draws for the energy and $1{,}850$ for the cut-edge count, and the frequencies below are quoted to the precision this supports. The chains forget their starts quickly. The share of each initial plan's district boundary still in place falls to its plateau of $0.20$ within $320$ to $520$ steps, and the plateau equals the share of one chain's boundary that a plan from another chain cuts, $0.20$, so a burn-in of $16{,}000$ steps is conservative by more than an order of magnitude. The agreement extends to the readouts reported below, which are computed per chain before pooling: the share of plans in which a given station is open differs between two chains by at most $0.08$ (mean absolute difference $0.02$ over stations), the share in which two open stations share a Group by at most $0.11$ (mean $0.003$ over pairs), and the contestedness of the LSOAs correlates at $0.96$ or better between any two chains. The Monte Carlo standard error of a station's open frequency, estimated from the four chains, is at most $0.016$, and $0.007$ for the median station.

Holding the state on rejection reweights the sampled law by the expected holding time, $1/a(s)$ for a state $s$ from which a proposal closes with probability $a(s)$ (Section~\ref{sec:convergence-status}). On this instance $a(s)$ is not constant. Pooling the post-burn-in steps, acceptance rises from $0.61$ in states with $46$ open stations to $0.73$ in states with $58$, because a plan with fewer and larger supernodes is harder to re-partition, so the expected holding time falls from $1.64$ steps to $1.37$ and the time law tilts toward plans with fewer open stations; the rejection probability is $0.334$ after a rejected step against $0.315$ after an accepted one. The tilt is small in its effects. Weighting each stored plan by the inverse length of the visit it belongs to, which recovers the law of the jump chain, moves the mean number of open stations from $52.48$ to $52.54$, the capacity shares by at most $0.002$ and the open frequency of any station by at most $0.008$ (mean $0.002$); one station at the threshold, Chiswick, moves from $0.894$ to $0.900$ and joins the set open in more than $90\%$ of plans. We report the time law, which is what the chain samples, and note that the retry budget $M$ enters it through $a(s)$.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/las/fig_traces.png}
\caption{The four London chains: traces of the energy, the number of stations and the number of cut edges, with the burn-in shaded, and the post-burn-in distributions by chain.}
\label{fig:las-traces}
\end{figure}

\subsubsection{What most sampled plans have in common}\label{sec:las-structure}
Every frequency in this section and the next is an estimate of a probability under $\pi_{\mathrm{FalCom}}$, the stationary law of the chain of Section~\ref{sec:falcom} as implemented, with its rejection rule and retry budget. That law is characterized only empirically (Section~\ref{sec:exp-enumeration}). It is supported on plans that satisfy the windows, it is not uniform over them, and on the enumerable instances it weights feasible plans by factors of up to several thousand, through spanning-tree counts, the recursion's capacity choices and the holding times just described. The readouts therefore come in three kinds. Statements about the benchmark, its infeasibility, its loads and its position relative to every sampled plan, do not depend on the law. Frequencies, how often an edge is cut, a station open or two stations grouped, do, and the replications of Section~\ref{sec:las-price} show which of them are stable under a change of tolerance and of demand proxy. Group readouts depend in addition on the level-2 construction, the greedy sequential recursion of Section~\ref{sec:recursive}, whose law is validated against enumeration only on small instances.

The ensemble separates the geography that most sampled plans reproduce from the choices that remain open. Over the $6{,}400$ plans, no edge of $G^1$ is cut by a station boundary in more than $44\%$ of them. $12.1\%$ of the $14{,}550$ edges are cut in fewer than $10\%$ of plans and lie inside catchments in nearly every plan; the other $87.9\%$ are contested. Group boundaries are more constrained. $30.8\%$ of edges are interior at level $2$, no edge is cut in more than $34\%$ of plans, and the level-2 boundary frequency concentrates on a few corridors in inner north London and along the river that separate clusters of stations (Figure~\ref{fig:las-boundary}). Seen from the units, an LSOA is served by its most frequent station in $48.3\%$ of plans on average ($48.5\%$ weighted by demand). Only $17$ LSOAs, $0.3\%$, keep one station in more than $90\%$ of plans, and these are the host LSOAs of the $17$ stations open in more than $90\%$ of plans, which constraint~\eqref{cons:anchor} ties to their station; $58.9\%$ of LSOAs have no station that serves them in even half of the plans. The map of contestedness shows the pattern: a settled core around each station, contested ground between stations, and settled fringes where stations are scarce (Heathrow, Havering, the southern edge), which are settled by scarcity rather than by geography.

\begin{figure}[!htbp]
\centering
\begin{subfigure}[t]{\linewidth}
  \centering
  \includegraphics[width=\linewidth]{figures/las/fig_boundary_freq.png}
  \caption{Boundary-cut frequency at level 1 (left) and level 2 (right).}
  \label{fig:las-boundary-panels}
\end{subfigure}

\begin{subfigure}[t]{0.55\linewidth}
  \centering
  \includegraphics[width=\linewidth]{figures/las/fig_contested.png}
  \caption{Contested LSOAs.}
  \label{fig:las-contested}
\end{subfigure}
\caption{Boundaries and contested areas in the pooled ensemble. (\subref{fig:las-boundary-panels}) Share of plans in which each edge of $G^1$ is cut by a station boundary (left) or a Group boundary (right); darker and thicker is more often cut, and the colour scale runs to the largest observed frequency. (\subref{fig:las-contested}) One minus the share of plans in which an LSOA is served by its most frequent station. Stations in cyan.}
\label{fig:las-boundary}
\end{figure}

Because the candidates are the real stations, the ensemble also describes each station (Figure~\ref{fig:las-stations}). A plan opens $52.5$ stations on average, between $43$ and $61$, and between $49$ and $56$ in the central $90\%$ of plans. Seventeen stations are open in more than $90\%$ of plans, led by Fulham ($97\%$), St John's Wood ($96\%$), Ilford, Deptford and Barnehurst ($94\%$ each) and Whipps Cross ($93\%$). No station is open in fewer than $40\%$ of plans, and the median station is open in $83\%$. The least used sites are those whose surroundings cannot fill a unit, or that sit next to a stronger neighbour: Chase Farm ($40\%$), Heathrow Airport ($41\%$), Twickenham ($49\%$), New Addington and Wembley ($60\%$), and Tolworth, Barnet and Richmond ($61\%$). An open frequency counts how often the law opens a station. It says nothing about the cost of closing one, which the ensemble does not measure. Capacity follows density. Over all open stations, $42\%$ carry one unit, $31\%$ two and $27\%$ three. Inner stations such as St John's Wood, Ilford, Camden, Deptford and Brent carry three units in $37\%$ to $42\%$ of the plans in which they are open, and outer stations such as Tolworth, Richmond, Chase Farm, Heathrow Airport and New Addington carry one unit in $67\%$ to $77\%$ of theirs.

The sampled Groups recover part of the operational structure (Figure~\ref{fig:las-groups}). A plan has $22.3$ Groups on average, between $20$ and $24$ in the central $90\%$ of plans, against the $21$ real Groups. We measure co-membership as the share of plans in which two stations lie in the same superdistrict, among the plans in which both are open. Two stations of the same real Group share a superdistrict in $27\%$ of such plans, two stations of different Groups in $2\%$. Part of this contrast is distance, since stations far apart never share a superdistrict. The adjusted Rand index between a sampled grouping of the open stations and the real Groups is $0.28$ on average, between $0.17$ and $0.39$ in the central $90\%$ of plans and $0.28$ among the $1{,}255$ plans with exactly $21$ Groups; between two sampled plans from different chains it is $0.19$. The real Groups are thus recovered somewhat better than one sampled plan recovers another. They are one contiguous, capacity-feasible grouping among many that the geography admits, a typical member of the family rather than its mode. The pairs the ensemble keeps together most often inside a real Group are Hornchurch and Romford ($67\%$), Barnehurst and Woolwich ($55\%$), Becontree and Hornchurch ($52\%$), and Oval and Waterloo and Ponders End and Chase Farm ($49\%$ each). Equally strong are pairs that the real Groups separate: Barnehurst and St Pauls Cray ($49\%$, Greenwich and Bromley Groups), Richmond and Putney ($48\%$), Fulham and Westminster ($47\%$), Waterloo and Deptford ($42\%$) and Heathrow Airport and Feltham ($41\%$). Fourteen cross-Group pairs exceed $30\%$ against $29$ within-Group pairs. The cross-Group pairs are the associations a planner would examine first, since they mark Group boundaries that balanced plans tend not to draw.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/las/fig_stations.png}
\caption{Top: share of plans in which each station is open, sorted. Bottom: capacity mix of each station in the plans in which it is open.}
\label{fig:las-stations}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.7\linewidth]{figures/las/fig_groups.png}
\caption{Share of plans in which two stations belong to the same superdistrict, among the plans in which both are open; stations ordered by their real Group, with white lines separating the $21$ Groups and the colour scale running to the largest off-diagonal value.}
\label{fig:las-groups}
\end{figure}

\subsubsection{An upper bound on the price of workload balance}\label{sec:las-price}
Figure~\ref{fig:las-slas} places the operational layout in the ensemble. Its energy is $7.53$ million demand-weighted minutes, $3.08$ million on the station leg and $4.45$ million on the Group leg. That is below every one of the $6{,}400$ stored plans, whose lowest energy is $8.71$ million, below the lowest energy any chain visited after burn-in, $8.48$ million, and $22\%$ below the ensemble mean of $9.68$ million (central $90\%$ of plans between $9.21$ and $10.19$ million). Of the $2.15$ million gap to the mean, $1.13$ million is on the station leg and $1.02$ million on the Group leg. The benchmark's Group leg is computed at the documented headquarters and the ensemble's at the $1$-median of each superdistrict; computing the benchmark at its $1$-medians would widen the gap. The station-leg metrics agree with the station leg. The operational layout has a demand-weighted mean drive time of $2.9$ minutes against $4.0$ in the ensemble ($3.8$ to $4.2$), a $90$th percentile of $4.8$ minutes against $7.0$ ($6.5$ to $7.6$), and $99.5\%$ of demand within eight minutes' drive of its catchment station against $94.5\%$ ($91.8\%$ to $96.7\%$).

The source of the gap is the station loads. The operational layout opens all $66$ stations, where the sampled plans open $43$ to $61$, and its loads range from $0.19$ to $4.68$ units, a factor of $25$. Measured against the widest windows the chain can produce (Proposition~\ref{prop:effective}: $0.72$ to $1.32$ units for a one-unit station, $1.51$ to $2.56$ for two units and $2.29$ to $3.79$ for three), $20$ of the $66$ stations fall outside. Ten serve less than $0.72$ units, among them Waterloo at $0.19$ and Smithfield and Heathrow Airport at about $0.5$; Deptford, at $4.68$, exceeds the three-unit range; and nine fall between the one-unit and two-unit ranges, loads that blocks of three ambulances cannot express. Against the nominal $\pm15\%$ windows of the model the count is $36$. None of these statements depends on the sampled law.

The ensemble is a sample from a law supported on plans that satisfy these windows, so its best plan is feasible and the minimum over the feasible set lies at or below it. The distance from that plan to the operational layout is therefore an upper bound on the access cost of workload balance at this calibration, block granularity included. Against the best stored plan the bound is $1.18$ million demand-weighted minutes, $16\%$ of the layout's energy, of which $0.86$ million is on the station leg. In the station-leg metrics, taking for each metric the best plan in the sample, the bound is $0.6$ minutes of mean drive time ($3.55$ against $2.92$), $1.2$ minutes at the $90$th percentile ($5.94$ against $4.78$) and $0.7$ points of eight-minute share ($98.8\%$ against $99.5\%$). Against a typical plan the differences are $2.15$ million, $1.1$ minutes and five points. The bound from the sample minimum is loose, since the chain does not seek low energy and Section~\ref{sec:exp-milp} shows that even FalCom's annealed states stay several percent above the optimum. %% PHASE2-PARAGRAPH
Part of this price is a limit of the demand proxy rather than of the layout. The smallest catchments sit on daytime hot spots, the West End, the City and the airport, which residential apportionment does not see. All of these are drive-time figures of a static assignment, catchment geometry rather than response performance, and the published response-time clock, which includes call handling and mobilisation, offers no validation of them. Whether the rest of the price is worth paying is an operational question that the ensemble informs but does not settle.

%% PHASE3-PARAGRAPH

The result does not depend on the tolerance. We repeat the study with $\varepsilon^1 = \varepsilon^2 = 0.25$, the widest tolerance that keeps the recursion's debt windows disjoint with margin (Claim~\ref{claim:app-unique} allows up to $2/5$), with four chains of $40{,}000$ steps and everything else unchanged. Acceptance rises to $70.0\%$ to $70.6\%$, the chains forget their starts in $400$ to $520$ steps, and the plans open $53.3$ stations on average ($47$ to $62$), $21$ of them in more than $90\%$ of plans. The run is half as long and its diagnostics are correspondingly weaker: the largest cross-chain Kolmogorov--Smirnov distance is $0.054$ for the energy, $0.053$ for the number of stations and $0.098$ for the cut-edge count, against within-chain references of $0.029$ to $0.045$, $0.024$ to $0.107$ and $0.051$ to $0.105$; split-$\hat R$ is at most $1.008$; the effective sample sizes per chain are $252$ to $438$ for the energy and $145$ to $285$ for the cut-edge count, so the $3{,}200$ stored plans are worth about $1{,}390$ and $800$ independent draws; and three of the four initial plans needed two attempts. The ensemble energy is $9.68$ million (central $90\%$ between $9.20$ and $10.20$ million, lowest stored plan $8.61$ million), within $0.1\%$ of its value at $\varepsilon = 0.15$; the mean drive time is $4.0$ minutes ($3.8$ to $4.2$) and the eight-minute share $94.4\%$ ($91.6\%$ to $96.7\%$). The operational layout still lies below every sampled plan. The wider windows admit $14$ of its $20$ out-of-window stations, but six remain outside, five below $0.56$ units and Deptford above $4.38$ (Proposition~\ref{prop:effective} at $\varepsilon = 0.25$); against the nominal $\pm25\%$ windows the count is $26$. The gap comes from the capacity model itself, in which every open station serves between $0.56$ and $4.4$ units in blocks of three ambulances, whereas the operational layout runs stations from $0.19$ to $4.68$ units. A calibration that lets a plan keep both its smallest and its largest stations needs a wider capacity range, which by Corollary~\ref{cor:app-invariant} means a larger block or the rejection-based treatment of $c \ge 4$. We leave it to future work.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/las/fig_slas.png}
\caption{The operational layout $s_{\mathrm{LAS}}$ (red line) against the ensemble: energy, demand-weighted mean drive time, $90$th-percentile drive time and the share of demand within eight minutes' drive of its catchment station.}
\label{fig:las-slas}
\end{figure}

"""
start = s.index("\\subsection{London Ambulance Service Case Study}")
end = s.index("\\section{Discussion and Conclusions}")
fb = s.rfind("\\FloatBarrier", start, end); assert fb > start
s = s[:start] + NEW + s[fb:]

# ---- Discussion ----
rep("On the London Ambulance Service, run on the $66$ real stations, independently started chains agree and forget their starts within a few hundred steps. The ensemble locates the contested and the settled service boundaries, the stations every balanced plan keeps and those it can do without, and the price in access cost that workload balance exacts from the operational layout.",
    "On the London Ambulance Service, run on the $66$ real stations, four chains started from different seeds agree on every observable to within the noise of a single chain and forget their starts within a few hundred steps. The ensemble locates the contested and the settled service boundaries, the stations that nearly every sampled plan opens and those that many plans leave closed, and an upper bound, within a static assignment model, on the access cost that workload balance would exact from the operational layout.")
rep("its empirical law is independent of the start, and that law spreads mass across capacity mixes roughly uniformly while following the spanning-tree law within a mix.",
    "its empirical law is independent of the start to within a total-variation distance of $0.01$ to $0.02$, and that law spreads mass across capacity mixes roughly uniformly while following the spanning-tree law within a mix only loosely, at a total-variation distance of $0.11$ to $0.24$.")

# ---- Appendix: data assembly and benchmark ----
rep("The station roster is the $66$ active stations of the LAS estate, compiled from the station list on the LAS website; the closed station N4 Lee is excluded.",
    "The station roster is the $66$ active stations of the LAS estate, compiled from a published station list and cross-referenced with the freedom-of-information pages of the LAS website and with OpenStreetMap, with coordinates geocoded from postcodes; the closed station N4 Lee is excluded.")
rep("Demand is the 2010 to 2015 mean of the monthly incident totals that LAS publishes by borough~\\citep{las_incidents}, about $1.065$ million incidents a year,",
    "Demand is the 2010 to 2015 mean of the monthly incident totals that LAS publishes by borough~\\citep{las_incidents}, a London Datastore release covering June 2009 to July 2016 from which one mislabelled month is dropped, about $1.065$ million incidents a year,")
rep("For the $264$ station and LSOA pairs that the routing engine cannot connect, isolated fragments of the network, straight-line distance at the same speed is used.",
    "Four LSOAs, isolated fragments of the network, cannot be reached from any station by the routing engine (Hillingdon 031A, Newham 034A, Bromley 018F and Camden 026E; $264$ station and LSOA pairs, $0.1\\%$ of demand, none of them hosting a station); for these, straight-line distance at $25$ km/h is used.")
rep("LAS withheld the sector polygons under FOI Act s31(1)(a), but the 2025 list of departments~\\citep{las_departments2025}, which lists the stations of each Group, and the 2023 operational estate map~\\citep{las_estate2023} together define",
    "No vector boundaries of the sectors or Groups are published. The August 2023 operational estate map, a raster image that LAS released in response to Freedom of Information request 6055~\\citep{las_estate2023}, and the 2025 list of departments~\\citep{las_departments2025}, which lists the stations of each Group, together define")
P.write_text(s)

# ---- bibliography ----
B = pathlib.Path("src/ref_final.bib"); b = B.read_text()
old = """@misc{las_departments2025,
  author = {{London Ambulance Service NHS Trust}},
  title  = {List of departments and group headquarters},
  year   = {2025}
}
@misc{las_estate2023,
  author = {{London Ambulance Service NHS Trust}},
  title  = {Operational estate map},
  year   = {2023}
}
@misc{las_incidents,
  author = {{Greater London Authority} and {London Ambulance Service NHS Trust}},
  title  = {Monthly ambulance incidents by borough: ambulance call-out data by type of incident, June 2009 to July 2016},
  year   = {2016},
  howpublished = {London Datastore, Strategic Crime Analysis (SCAN) release},
  note   = {URL and access date to be added by the authors}
}"""
new = """@misc{las_departments2025,
  author = {{London Ambulance Service NHS Trust}},
  title  = {List of departments and group headquarters},
  year   = {2025},
  howpublished = {Document LAS-List-Of-Departments-2025, London Ambulance Service NHS Trust},
  note   = {URL and access date to be added by the authors}
}
@misc{las_estate2023,
  author = {{London Ambulance Service NHS Trust}},
  title  = {Operational estate map, August 2023},
  year   = {2024},
  howpublished = {Raster map released as the attachment to Freedom of Information request 6055, response dated 7 February 2024},
  note   = {URL and access date to be added by the authors}
}
@misc{las_incidents,
  author = {{Greater London Authority} and {London Ambulance Service NHS Trust}},
  title  = {Monthly ambulance incidents by borough: ambulance call-out data by type of incident, June 2009 to July 2016},
  year   = {2016},
  howpublished = {London Datastore, Greater London Authority},
  note   = {URL and access date to be added by the authors}
}"""
assert b.count(old) == 1; B.write_text(b.replace(old, new))
print("applied")
