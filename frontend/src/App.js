import { useState, useEffect } from "react";

const API = "http://localhost:5000";
const C = {
  bg:"#0a0e1a", panel:"#111827", card:"#1a2235", border:"#1f2d45",
  accent:"#00d4ff", green:"#10b981", yellow:"#f59e0b", red:"#ef4444",
  purple:"#7c3aed", text:"#e2e8f0", muted:"#64748b"
};

function Badge({text, color}){
  return <span style={{background:color+"22",color,border:`1px solid ${color}44`,borderRadius:4,padding:"2px 8px",fontSize:11,fontWeight:700}}>{text}</span>;
}

function StatCard({label,value,color=C.accent,unit="%"}){
  return(
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,padding:"14px 18px",flex:1,minWidth:110}}>
      <div style={{color:C.muted,fontSize:10,fontWeight:700,letterSpacing:1.5,textTransform:"uppercase",marginBottom:6}}>{label}</div>
      <div style={{color,fontSize:26,fontWeight:800,fontFamily:"monospace"}}>{value}{unit}</div>
    </div>
  );
}

function BarH({label,value,max,color,rank}){
  return(
    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
      {rank&&<span style={{width:20,color:C.muted,fontSize:11,textAlign:"right"}}>{rank}</span>}
      <div style={{width:160,fontSize:11,color:C.muted,textAlign:"right",fontFamily:"monospace"}}>{label}</div>
      <div style={{flex:1,height:16,background:C.border,borderRadius:4,overflow:"hidden"}}>
        <div style={{width:`${(value/max)*100}%`,height:"100%",background:`linear-gradient(90deg,${color},${color}88)`,borderRadius:4,transition:"width 1s ease"}}/>
      </div>
      <span style={{width:45,fontSize:11,color,fontFamily:"monospace",fontWeight:700}}>{(value*100).toFixed(1)}%</span>
    </div>
  );
}

// ── ONGLET TACHE 4 ──────────────────────────────────────────────────────────
function Tache4Panel(){
  const [tab,setTab]         = useState("q1");
  const [q1,setQ1]           = useState(null);
  const [q2,setQ2]           = useState(null);
  const [q3,setQ3]           = useState(null);
  const [q4,setQ4]           = useState(null);
  const [q5,setQ5]           = useState(null);
  const [loading,setLoading] = useState(false);

  useEffect(()=>{
    setLoading(true);
    Promise.all([
      fetch(`${API}/tache4/feature-importance`).then(r=>r.json()),
      fetch(`${API}/tache4/stability`).then(r=>r.json()),
      fetch(`${API}/tache4/confusion-matrix`).then(r=>r.json()),
      fetch(`${API}/tache4/bias-variance`).then(r=>r.json()),
      fetch(`${API}/tache4/rf-vs-dt`).then(r=>r.json()),
    ]).then(([d1,d2,d3,d4,d5])=>{
      setQ1(d1); setQ2(d2); setQ3(d3); setQ4(d4); setQ5(d5);
      setLoading(false);
    }).catch(()=>setLoading(false));
  },[]);

  const tabs=[["q1","Q1 — Features"],["q2","Q2 — Stabilite"],["q3","Q3 — Erreurs"],["q4","Q4 — Biais/Variance"],["q5","Q5 — RF vs DT"]];

  if(loading) return <div style={{padding:40,textAlign:"center",color:C.muted}}>Chargement des analyses Tache 4...</div>;

  return(
    <div>
      {/* Sub-tabs */}
      <div style={{display:"flex",gap:4,marginBottom:20,flexWrap:"wrap"}}>
        {tabs.map(([key,label])=>(
          <button key={key} onClick={()=>setTab(key)} style={{
            background:tab===key?C.accent+"22":"transparent",
            color:tab===key?C.accent:C.muted,
            border:`1px solid ${tab===key?C.accent:C.border}`,
            borderRadius:8,padding:"8px 14px",cursor:"pointer",fontSize:12,
            fontWeight:700,fontFamily:"monospace"
          }}>{label}</button>
        ))}
      </div>

      {/* Q1 — Feature Importance */}
      {tab==="q1"&&q1&&(
        <div>
          <div style={{color:C.text,fontWeight:800,fontSize:15,marginBottom:6}}>Importance des Features (feature_importances_)</div>
          <div style={{color:C.muted,fontSize:12,marginBottom:16}}>Reduction moyenne d'impurete (Gini) sur tous les arbres de la foret</div>
          <div style={{display:"flex",gap:12,marginBottom:20,flexWrap:"wrap"}}>
            {q1.top3.map((f,i)=>(
              <div key={i} style={{background:C.card,border:`1px solid ${C.green}44`,borderRadius:10,padding:"14px 18px",flex:1,minWidth:140}}>
                <div style={{color:C.green,fontSize:10,fontWeight:700,letterSpacing:1.5,textTransform:"uppercase"}}>Top {i+1}</div>
                <div style={{color:C.text,fontWeight:800,fontSize:14,marginTop:4}}>{f.feature}</div>
                <div style={{color:C.green,fontSize:22,fontWeight:800,fontFamily:"monospace",marginTop:4}}>{(f.importance*100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
          <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
            <div style={{color:C.muted,fontSize:11,fontWeight:700,letterSpacing:1.5,textTransform:"uppercase",marginBottom:14}}>Toutes les features (18)</div>
            {q1.data.map((f,i)=>(
              <BarH key={i} rank={f.rank} label={f.feature} value={f.importance} max={0.18} color={i<3?C.green:i<6?C.accent:C.muted}/>
            ))}
          </div>
          <div style={{background:C.green+"11",border:`1px solid ${C.green}44`,borderRadius:10,padding:16,marginTop:16}}>
            <div style={{color:C.green,fontWeight:700,marginBottom:6}}>Interpretation</div>
            <div style={{color:C.text,fontSize:12,lineHeight:1.6}}>
              RADIUS_RATIO, ELONGATEDNESS et SCATTER_RATIO capturent les differences geometriques fondamentales entre un bus (grand, allonge) et une voiture compacte (saab/opel). Ces features sont coherentes avec la perception visuelle humaine des silhouettes de vehicules.
            </div>
          </div>
        </div>
      )}

      {/* Q2 — Stabilite */}
      {tab==="q2"&&q2&&(
        <div>
          <div style={{color:C.text,fontWeight:800,fontSize:15,marginBottom:6}}>Stabilite des Predictions (10 random_states)</div>
          <div style={{display:"flex",gap:12,marginBottom:20,flexWrap:"wrap"}}>
            <StatCard label="Moyenne" value={(q2.summary.mean*100).toFixed(2)} color={C.green}/>
            <StatCard label="Ecart-type" value={"±"+(q2.summary.std*100).toFixed(2)} color={C.accent} unit="%"/>
            <StatCard label="Min" value={(q2.summary.min*100).toFixed(2)} color={C.yellow}/>
            <StatCard label="Max" value={(q2.summary.max*100).toFixed(2)} color={C.purple}/>
          </div>
          <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
            <div style={{color:C.muted,fontSize:11,fontWeight:700,letterSpacing:1.5,textTransform:"uppercase",marginBottom:14}}>Accuracy par random_state</div>
            {q2.runs.map((r,i)=>(
              <div key={i} style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}>
                <span style={{width:90,color:C.muted,fontSize:11,fontFamily:"monospace"}}>seed={r.random_state}</span>
                <div style={{flex:1,height:14,background:C.border,borderRadius:3,overflow:"hidden"}}>
                  <div style={{width:`${r.test_accuracy*100}%`,height:"100%",background:C.accent,borderRadius:3}}/>
                </div>
                <span style={{width:50,fontFamily:"monospace",color:C.accent,fontSize:11,fontWeight:700}}>{(r.test_accuracy*100).toFixed(2)}%</span>
              </div>
            ))}
          </div>
          <div style={{background:C.green+"11",border:`1px solid ${C.green}44`,borderRadius:10,padding:16,marginTop:16}}>
            <div style={{color:C.green,fontWeight:700,marginBottom:6}}>Conclusion</div>
            <div style={{color:C.text,fontSize:12}}>Variabilite tres faible (std=0.83%). Le modele est ROBUSTE et ne depend pas de l'initialisation aleatoire. 100 arbres suffisent pour stabiliser les votes majoritaires.</div>
          </div>
        </div>
      )}

      {/* Q3 — Erreurs */}
      {tab==="q3"&&q3&&(
        <div>
          <div style={{color:C.text,fontWeight:800,fontSize:15,marginBottom:6}}>Analyse des Erreurs de Classification</div>
          <div style={{display:"flex",gap:12,marginBottom:20,flexWrap:"wrap"}}>
            <StatCard label="Erreurs" value={q3.n_errors} color={C.red} unit="/190"/>
            <StatCard label="Taux erreur" value={(q3.error_rate*100).toFixed(1)} color={C.yellow}/>
            <StatCard label="Taux succes" value={((1-q3.error_rate)*100).toFixed(1)} color={C.green}/>
          </div>
          {/* Matrice de confusion */}
          <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20,marginBottom:16}}>
            <div style={{color:C.muted,fontSize:11,fontWeight:700,letterSpacing:1.5,textTransform:"uppercase",marginBottom:14}}>Matrice de Confusion</div>
            <table style={{borderCollapse:"collapse",width:"100%"}}>
              <thead>
                <tr>
                  <th style={{padding:"6px 10px",color:C.muted,fontSize:11}}>Reel / Predit</th>
                  {q3.classes.map(c=><th key={c} style={{padding:"6px 10px",color:C.accent,fontSize:11,fontWeight:700}}>{c}</th>)}
                </tr>
              </thead>
              <tbody>
                {q3.matrix.map((row,ri)=>(
                  <tr key={ri}>
                    <td style={{padding:"8px 10px",color:C.accent,fontSize:11,fontWeight:700}}>{q3.classes[ri]}</td>
                    {row.map((val,ci)=>(
                      <td key={ci} style={{padding:"8px 14px",textAlign:"center",
                        background:ri===ci?"rgba(16,185,129,0.2)":val>0?"rgba(239,68,68,0.1)":"transparent",
                        border:`1px solid ${C.border}`,borderRadius:4,
                        fontFamily:"monospace",fontSize:14,fontWeight:ri===ci?800:400,
                        color:ri===ci?C.green:val>0?C.red:C.muted}}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Erreurs fréquentes */}
          <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
            <div style={{color:C.muted,fontSize:11,fontWeight:700,letterSpacing:1.5,textTransform:"uppercase",marginBottom:12}}>Paires d'erreurs les plus frequentes</div>
            {q3.error_pairs.map((ep,i)=>(
              <div key={i} style={{display:"flex",alignItems:"center",gap:10,marginBottom:8,background:C.panel,borderRadius:8,padding:"8px 14px"}}>
                <Badge text={ep.real} color={C.yellow}/>
                <span style={{color:C.muted}}>→</span>
                <Badge text={ep.predicted} color={C.red}/>
                <span style={{marginLeft:"auto",fontFamily:"monospace",color:C.red,fontWeight:700}}>{ep.count} cas</span>
              </div>
            ))}
          </div>
          <div style={{background:"rgba(239,68,68,0.08)",border:`1px solid ${C.red}44`,borderRadius:10,padding:16,marginTop:16}}>
            <div style={{color:C.red,fontWeight:700,marginBottom:6}}>Pattern identifie</div>
            <div style={{color:C.text,fontSize:12}}>Confusion principale : saab ↔ opel (silhouettes de berlines similaires). Bus et van rarement confondus avec les voitures. Les erreurs ont une confiance &lt; 60% (le modele hesite correctement).</div>
          </div>
        </div>
      )}

      {/* Q4 — Biais Variance */}
      {tab==="q4"&&q4&&(
        <div>
          <div style={{color:C.text,fontWeight:800,fontSize:15,marginBottom:6}}>Analyse Biais-Variance</div>
          <div style={{display:"flex",gap:12,marginBottom:20,flexWrap:"wrap"}}>
            <div style={{flex:1,background:C.red+"11",border:`1px solid ${C.red}44`,borderRadius:10,padding:14}}>
              <div style={{color:C.red,fontWeight:700,fontSize:12,marginBottom:4}}>Overfitting detecte</div>
              {q4.overfitting.map((r,i)=><div key={i} style={{color:C.text,fontSize:11}}>n={r.n_estimators}, d={r.max_depth||"None"} — Train:{(r.train_acc*100).toFixed(1)}% Test:{(r.test_acc*100).toFixed(1)}%</div>)}
            </div>
            <div style={{flex:1,background:C.yellow+"11",border:`1px solid ${C.yellow}44`,borderRadius:10,padding:14}}>
              <div style={{color:C.yellow,fontWeight:700,fontSize:12,marginBottom:4}}>Underfitting detecte</div>
              {q4.underfitting.map((r,i)=><div key={i} style={{color:C.text,fontSize:11}}>n={r.n_estimators}, d={r.max_depth||"None"} — Train:{(r.train_acc*100).toFixed(1)}% Test:{(r.test_acc*100).toFixed(1)}%</div>)}
            </div>
            <div style={{flex:1,background:C.green+"11",border:`1px solid ${C.green}44`,borderRadius:10,padding:14}}>
              <div style={{color:C.green,fontWeight:700,fontSize:12,marginBottom:4}}>Configuration optimale</div>
              {q4.optimal.map((r,i)=><div key={i} style={{color:C.text,fontSize:11}}>n={r.n_estimators}, d={r.max_depth} — Test:{(r.test_acc*100).toFixed(1)}%</div>)}
            </div>
          </div>
          <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,overflow:"hidden"}}>
            <table style={{width:"100%",borderCollapse:"collapse"}}>
              <thead>
                <tr style={{background:C.panel}}>
                  {["n_estimators","max_depth","Train Acc","Test Acc","Biais","Variance","Diagnostic"].map(h=>(
                    <th key={h} style={{padding:"10px 12px",textAlign:"left",color:C.muted,fontSize:10,fontWeight:700,letterSpacing:1,borderBottom:`1px solid ${C.border}`}}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {q4.data.map((r,i)=>{
                  const isOpt=r.diagnostic==="OPTIMAL";
                  const isOvf=r.diagnostic.includes("Overfitting");
                  const isUnf=r.diagnostic.includes("Underfitting");
                  const bg=isOpt?C.green+"11":isOvf?C.red+"08":isUnf?C.yellow+"08":"transparent";
                  return(
                    <tr key={i} style={{borderBottom:`1px solid ${C.border}`,background:bg}}>
                      <td style={{padding:"8px 12px",fontFamily:"monospace",fontSize:12}}>{r.n_estimators}</td>
                      <td style={{padding:"8px 12px",fontFamily:"monospace",fontSize:12}}>{r.max_depth||"None"}</td>
                      <td style={{padding:"8px 12px",fontFamily:"monospace",fontSize:12}}>{(r.train_acc*100).toFixed(2)}%</td>
                      <td style={{padding:"8px 12px",fontFamily:"monospace",color:isOpt?C.green:isOvf?C.red:C.text,fontWeight:isOpt?800:400,fontSize:12}}>{(r.test_acc*100).toFixed(2)}%</td>
                      <td style={{padding:"8px 12px",fontFamily:"monospace",fontSize:12,color:r.biais>0.2?C.yellow:C.text}}>{r.biais.toFixed(4)}</td>
                      <td style={{padding:"8px 12px",fontFamily:"monospace",fontSize:12,color:r.variance>0.15?C.red:C.text}}>{r.variance.toFixed(4)}</td>
                      <td style={{padding:"8px 12px"}}><Badge text={r.diagnostic} color={isOpt?C.green:isOvf?C.red:isUnf?C.yellow:C.muted}/></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Q5 — RF vs DT */}
      {tab==="q5"&&q5&&(
        <div>
          <div style={{color:C.text,fontWeight:800,fontSize:15,marginBottom:6}}>Random Forest vs Arbre de Decision</div>
          <div style={{display:"flex",gap:12,marginBottom:20,flexWrap:"wrap"}}>
            <StatCard label="Avantage Accuracy" value={"+"+(q5.advantage_rf*100).toFixed(2)} color={C.green}/>
            <StatCard label="Avantage CV Score" value={"+"+(q5.advantage_cv*100).toFixed(2)} color={C.accent}/>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
            <div style={{background:C.green+"11",border:`1px solid ${C.green}44`,borderRadius:12,padding:20}}>
              <div style={{color:C.green,fontWeight:800,fontSize:14,marginBottom:14}}>Random Forest</div>
              {q5.random_forest.map((r,i)=>(
                <div key={i} style={{marginBottom:12}}>
                  <div style={{color:C.text,fontSize:12,fontWeight:600,marginBottom:4}}>{r.config}</div>
                  <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                    <Badge text={"Test: "+(r.test_acc*100).toFixed(2)+"%"} color={C.green}/>
                    <Badge text={"CV: "+(r.cv_score*100).toFixed(2)+"%"} color={C.accent}/>
                  </div>
                </div>
              ))}
            </div>
            <div style={{background:C.yellow+"11",border:`1px solid ${C.yellow}44`,borderRadius:12,padding:20}}>
              <div style={{color:C.yellow,fontWeight:800,fontSize:14,marginBottom:14}}>Arbre de Decision</div>
              {q5.decision_tree.map((r,i)=>(
                <div key={i} style={{marginBottom:10}}>
                  <div style={{color:C.text,fontSize:11,fontWeight:600,marginBottom:3}}>{r.config}</div>
                  <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                    <Badge text={"Test: "+(r.test_acc*100).toFixed(2)+"%"} color={C.yellow}/>
                    <Badge text={"Var: "+(r.variance*100).toFixed(1)+"%"} color={r.variance>0.15?C.red:C.muted}/>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div style={{background:C.green+"11",border:`1px solid ${C.green}44`,borderRadius:10,padding:16,marginTop:16}}>
            <div style={{color:C.green,fontWeight:700,marginBottom:6}}>Verdict</div>
            <div style={{color:C.text,fontSize:12}}>Random Forest superieur sur tous les criteres de performance : +3.16% accuracy, +6.48% CV score, variance 2.5x plus faible. Le DT reste utile uniquement pour l'interpretabilite (regles metier explicites).</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── APP PRINCIPALE ──────────────────────────────────────────────────────────
export default function App(){
  const [mainTab, setMainTab] = useState("predict");

  const TABS=[
    {key:"predict",label:"Prediction"},
    {key:"tache4", label:"Tache 4 — Analyse RF"},
  ];

  return(
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,fontFamily:"'IBM Plex Mono',monospace",fontSize:13}}>
      {/* Header */}
      <div style={{background:C.panel,borderBottom:`1px solid ${C.border}`,padding:"0 24px",display:"flex",alignItems:"center",gap:16,height:56}}>
        <div style={{fontWeight:800,fontSize:15,letterSpacing:2,color:C.accent}}>ML<span style={{color:C.text}}>STUDIO</span></div>
        <div style={{color:C.border,fontSize:20}}>|</div>
        <div style={{color:C.muted,fontSize:11}}>Projet 16 — Vehicle Classification</div>
        <div style={{marginLeft:"auto",display:"flex",gap:8}}>
          <Badge text="TACHE 4" color={C.purple}/>
          <Badge text="Random Forest" color={C.green}/>
        </div>
      </div>
      {/* Tabs */}
      <div style={{background:C.panel,borderBottom:`1px solid ${C.border}`,padding:"0 24px",display:"flex",gap:4}}>
        {TABS.map(t=>(
          <button key={t.key} onClick={()=>setMainTab(t.key)} style={{
            background:"transparent",border:"none",
            borderBottom:mainTab===t.key?`2px solid ${C.accent}`:"2px solid transparent",
            color:mainTab===t.key?C.accent:C.muted,
            padding:"14px 16px",cursor:"pointer",fontSize:12,fontWeight:700,fontFamily:"inherit"
          }}>{t.label}</button>
        ))}
      </div>
      <div style={{padding:24,maxWidth:1100,margin:"0 auto"}}>
        {mainTab==="predict"&&(
          <div style={{textAlign:"center",padding:60,color:C.muted}}>
            <div style={{fontSize:40}}>🚗</div>
            <div style={{marginTop:12,fontSize:14}}>Interface de prediction disponible dans l'onglet ml-dashboard (Tache 2).</div>
            <div style={{marginTop:8,fontSize:12}}>Cette vue est dediee a l'analyse Tache 4.</div>
          </div>
        )}
        {mainTab==="tache4"&&<Tache4Panel/>}
      </div>
    </div>
  );
}
