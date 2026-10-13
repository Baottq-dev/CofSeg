// ws_flower.js — thẻ Hoa: gán mask hoa trong từng tán.
// Server giữ mask (nét người ⊕ nền máy); client chỉ gửi thao tác và vẽ lại
// hai lớp phủ server trả về (mask hoa, nét của người) đúng chỗ khung tán.
const FL_TOOLS={hoa:{label:1,color:'#22c55e',name:'Hoa'},khong:{label:0,color:'#ef4444',name:'Không hoa'},
 boqua:{label:255,color:'#94a3b8',name:'Bỏ qua'},tay:{label:'erase',color:'#e2e8f0',name:'Tẩy'}};
let flRounds=[],flRound=lsGet('cofseg_flround',''),flCrowns=[],flCrownsFor=null,flCrown=-1,flTool='hoa';
let flBrush=lsGet('cofseg_flbrush',12),flOpacity=lsGet('cofseg_flopacity',0.8);
let flLayers=Object.assign({mask:true,human:true,others:true},lsGet('cofseg_fllayers',{}));
let flImgs={},flStroke=null,flMouse=null,flSpace=false,flBusy=false,flTok=0,flCfgTok=0;
// Công cụ Kéo ảnh (H): chuột trái kéo là di chuyển ảnh; bấm không kéo lên một
// tán là mở tán đó. flPanAt = chỗ bắt đầu kéo, để phân biệt bấm với kéo.
let flPanAt=null;
let flQueue=null,flPending=null,flSnap=false,flQTok=0;
// Đồng hồ theo tán: chỉ đếm khi có thao tác trong 60 giây gần nhất.
let flSecs={},flLast=0,flTick=null;
function flKey(k){return current+'#'+k;}
function flTouch(){flLast=Date.now();}
function flTimerTick(){if(wsActive!=='flower'||flCrown<0){$('stTimer').textContent='';return;}
 const k=flKey(flCrown);if(Date.now()-flLast<60000)flSecs[k]=(flSecs[k]||0)+1;
 const s=Math.round(flSecs[k]||0);$('stTimer').textContent='⏱ '+String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
 flRenderMeta();}
function flRoundObj(){return flRounds.find(r=>r.id===flRound);}
function flCur(){return flCrowns.find(c=>c.crown===flCrown);}
async function flApi(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 let j=null;try{j=await r.json();}catch(e){j={ok:false,error:'HTTP '+r.status};}
 if(!r.ok||!j.ok)throw new Error(j.error||('HTTP '+r.status));return j;}
function flErr(e){$('hint').textContent='⚠ '+(e.message||e);svst('err','⚠ thẻ Hoa',String(e.message||e));}
// --- vòng -----------------------------------------------------------------
async function flLoadConfig(){const tok=++flCfgTok;let d=null;
 try{d=await(await fetch('/api/fl/config',{cache:'no-store'})).json();}catch(e){d={rounds:[],errors:[String(e)]};}
 if(tok!==flCfgTok)return;
 flRounds=d.rounds||[];if(!flRoundObj()&&flRounds.length)flRound=flRounds[0].id;
 const s=$('flRound');s.innerHTML='';
 flRounds.forEach(r=>{const o=document.createElement('option');o.value=r.id;o.textContent=r.label+' · '+({scribble:'nét vẽ + máy học',manual:'vẽ tay',review:'duyệt'}[r.kind]||r.kind);s.appendChild(o);});
 s.value=flRound;
 const r=flRoundObj();
 $('flRoundInfo').textContent=(d.errors&&d.errors.length?'⚠ '+d.errors.join(' · ')+' · ':'')+
  (r?((r.pick&&r.pick.fields)?'Ảnh của '+r.pick.fields.join(', ')+' · ':'')+'nhãn lưu ở '+(d.data_dir||''):'Chưa có vòng nào trong configs/flower_rounds.yaml');
 $('flLearn').hidden=!(r&&r.kind==='scribble');flRenderQueue();}
// --- tán của ảnh ------------------------------------------------------------
async function flLoadCrowns(){if(!current||!flRound)return;const who=current,tok=++flTok;
 let d=null;try{d=await(await fetch('/api/fl/crowns?name='+encodeURIComponent(who)+'&round='+encodeURIComponent(flRound)+'&who='+encodeURIComponent(annot),{cache:'no-store'})).json();}catch(e){d=null;}
 if(tok!==flTok||who!==current)return;
 flCrowns=(d&&d.crowns)||[];flCrownsFor=who;
 if(flCrown>=0&&!flCur())flCrown=-1;
 flRender();if(flCrown>=0)flFetchLayers();draw();
 // N / hàng đợi vừa đổi ảnh: mở luôn tán đang chờ khi ảnh đã nạp xong.
 if(flPending&&flPending.name===who){const k=flPending.crown;flPending=null;flOpen(k);}}
function flHit(x,y){let best=-1,ba=Infinity;
 flCrowns.forEach(c=>{if(c.area<ba&&pip(c.points,x,y)){best=c.crown;ba=c.area;}});return best;}
function flFitCrown(){const c=flCur();if(!c||!img.width)return;
 const [x0,y0,x1,y1]=c.bbox,bw=x1-x0,bh=y1-y0;
 const s=Math.min(cv.width/(bw*1.25),cv.height/(bh*1.25));
 zoom=Math.min(12,Math.max(0.1,s/baseFit));scale=baseFit*zoom;
 ox=cv.width/2-(x0+bw/2)*scale;oy=cv.height/2-(y0+bh/2)*scale;setZlbl();draw();}
function flOpen(k){const c0=flCrowns.find(c=>c.crown===k);
 if(c0&&c0.in_round===false){$('hint').textContent='⚠ Tán #'+(k+1)+' không thuộc phần việc của người '+annot+' trong vòng này.';return;}
 flCrown=k;flImgs={};flStroke=null;flTouch();$('flLearnInfo').textContent='';flRenderQueue();
 flFitCrown();flRender();flFetchLayers();
 $('hint').textContent=(flRoundObj()&&flRoundObj().kind==='scribble')
  ?'Vẽ vài nét Hoa (1) lên cụm hoa và Không hoa (2) lên thứ dễ nhầm (lá vàng nắng, chói, đất sáng), rồi Học & tô (T). Sai chỗ nào vẽ thêm nét ở đó.'
  :'Tô hoa bằng cọ. Phần trong tán chưa tô sẽ thành Không hoa khi bấm Xong (Enter).';}
function flClose(){flCrown=-1;flImgs={};flStroke=null;flRender();draw();
 $('hint').textContent='Bấm vào một tán để bắt đầu gán hoa · N mở tán chưa xong kế tiếp.';}
// Hai lớp phủ PNG cắt theo khung tán; X-Crop nói đặt ở đâu trên ảnh.
async function flFetchLayers(){const who=current,k=flCrown;if(!who||k<0)return;
 const get=async layer=>{const r=await fetch('/api/fl/view?name='+encodeURIComponent(who)+'&crown='+k+'&layer='+layer
  +'&round='+encodeURIComponent(flRound)+'&who='+encodeURIComponent(annot)+'&t='+Date.now(),{cache:'no-store'});
  if(!r.ok)return null;const box=(r.headers.get('X-Crop')||'').split(',').map(Number);
  const bmp=await createImageBitmap(await r.blob());return {bmp,box};};
 try{const [m,h]=await Promise.all([get('mask'),get('human')]);
  if(who!==current||k!==flCrown)return;flImgs={mask:m,human:h};draw();}catch(e){}}
// --- thao tác ---------------------------------------------------------------
function flBody(extra){return Object.assign({round:flRound,name:current,crown:flCrown,who:annot},extra||{});}
function flApply(j){const c=flCur();if(c){c.n_ops=j.n_ops;c.cover=j.cover;if(j.state)c.state=j.state;else if(c.state==='todo'&&j.n_ops)c.state='doing';}
 flRender();flFetchLayers();svst('ok','✔ đã lưu hoa '+hhmm());}
async function flSendStroke(s){if(!s.pts.length)return;
 const t=FL_TOOLS[s.tool];
 try{flApply(await flApi('/api/fl/stroke',flBody({label:t.label,r:s.r,snap:!!s.snap,pts:s.pts.map(p=>[Math.round(p[0]*10)/10,Math.round(p[1]*10)/10])})));}
 catch(e){flErr(e);draw();}}
async function flUndo(){if(flCrown<0||flBusy)return;flTouch();
 try{flApply(await flApi('/api/fl/undo',flBody()));}catch(e){flErr(e);}}
async function flClassify(){const r=flRoundObj();if(flCrown<0||flBusy||!r||r.kind!=='scribble')return;flTouch();
 flBusy=true;$('flLearn').disabled=true;$('hint').textContent='⏳ Đang học từ nét của cả vòng rồi tô tán này…';
 try{const j=await flApi('/api/fl/classify',flBody());flApply(j);
  const n=x=>Number(x).toLocaleString('vi-VN',{maximumFractionDigits:1});
  $('flLearnInfo').textContent='Học từ '+n(j.n_samples)+' pixel của '+j.n_crowns+' tán · đặc trưng '+n(j.t_feat)+' s · học '+n(j.t_fit)+' s · tô '+n(j.t_pred)+' s';
  $('hint').textContent='Máy đã tô (vàng). Chỗ nào sai thì vẽ thêm nét đúng nhãn ở đó rồi Học & tô lại; ổn thì Xong (Enter).';}
 catch(e){flErr(e);}
 finally{flBusy=false;$('flLearn').disabled=false;}}
async function flFinish(){if(flCrown<0||flBusy)return;const k=flCrown,key=flKey(k);
 try{const j=await flApi('/api/fl/finish',flBody({seconds:Math.round(flSecs[key]||0)}));
  delete flSecs[key];flApply(j);
  if(flQueue){const q=flQueue.items.find(x=>x.name===current&&x.crown===k);if(q)q.state='done';flLoadQueue();}
  flNext();}catch(e){flErr(e);}}
// N: vòng đã dựng thì theo hàng đợi (có thể sang ảnh khác); vòng chọn tay
// thì tán chưa xong kế tiếp trong ảnh đang mở.
function flNext(){if(flQueue&&flQueue.items.length){const it=flQueue.items;
  const i0=it.findIndex(x=>x.name===current&&x.crown===flCrown);
  for(let d=1;d<=it.length;d++){const x=it[(Math.max(i0,-1)+d+it.length)%it.length];
   if(!FL_DONE.includes(x.state)&&!(x.name===current&&x.crown===flCrown)){flGo(x);return;}}
  flClose();$('hint').textContent='✔ Phần việc của người '+annot+' trong vòng này đã xong hết.';return;}
 if(!flCrowns.length)return;
 const order=flCrowns.map(c=>c.crown),i0=Math.max(0,order.indexOf(flCrown));
 for(let d=1;d<=order.length;d++){const c=flCrowns[(i0+d)%order.length];
  if(!FL_DONE.includes(c.state)&&c.crown!==flCrown&&c.in_round!==false){flOpen(c.crown);return;}}
 flClose();$('hint').textContent='✔ Ảnh này đã xong mọi tán · PgDn để sang ảnh sau.';}
function setFlTool(t){flTool=t;document.querySelectorAll('#flTools button').forEach(b=>{const on=b.dataset.tool===t;b.classList.toggle('on',on);b.setAttribute('aria-checked',on);});
 if(flMouse)flCursor(flMouse[0],flMouse[1]);draw();}
function setFlBrush(px){flBrush=Math.round(Math.max(2,Math.min(80,px)));lsSet('cofseg_flbrush',flBrush);
 $('flBrush').value=flBrush;$('flBrushLbl').textContent=flBrush+' px';draw();}
function setFlOpacity(v){flOpacity=Math.max(0,Math.min(1,v));lsSet('cofseg_flopacity',flOpacity);$('flOpacity').value=Math.round(flOpacity*100);draw();}
function toggleFlLayer(l){flLayers[l]=!flLayers[l];lsSet('cofseg_fllayers',flLayers);flRenderLayers();draw();}
// --- cột chi tiết -------------------------------------------------------------
const FL_STATE={todo:'chưa',doing:'đang làm',done:'xong',accepted:'đã nhận',fixing:'đang sửa',fixed:'đã sửa',auto:'tự nhận'};
const FL_DONE=['done','accepted','fixed','auto'];
function flRenderLayers(){document.querySelectorAll('.fllayers button').forEach(b=>{const on=!!flLayers[b.dataset.layer];b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});}
function flRenderMeta(){const c=flCur();if(!c)return;const s=Math.round(flSecs[flKey(c.crown)]||0);
 $('flMeta').textContent=c.n_ops+' thao tác · '+Math.round(c.area).toLocaleString('vi-VN')+' px'+(c.seconds?(' · đã tính '+Math.round(c.seconds)+' s'):'')+(s?(' · phiên này '+s+' s'):'');}
function flRender(){const c=flCur();$('flCrownBox').hidden=!c;$('flCount').textContent=flCrowns.length;
 // Vòng duyệt: Chấp nhận / Sửa trước; Xong chỉ hiện khi đang sửa.
 const rv=flRoundObj()&&flRoundObj().kind==='review';
 $('flReviewBtns').hidden=!(rv&&c&&!['accepted','fixed','fixing'].includes(c.state));
 $('flDone').hidden=!!(rv&&!(c&&c.state==='fixing'));
 if(c){$('flTitle').textContent='Tán #'+(c.crown+1);
  const st=$('flState');st.textContent=FL_STATE[c.state]||c.state;st.className='pchip'+(FL_DONE.includes(c.state)?' tp':((c.state==='doing'||c.state==='fixing')?' fp':''));
  $('flCover').textContent=(c.n_ops||c.cover)?(100*c.cover).toLocaleString('vi-VN',{maximumFractionDigits:1})+' %':'—';flRenderMeta();}
 const box=$('flChips');box.innerHTML='';
 flCrowns.forEach(x=>{const b=document.createElement('button');b.className='flchip '+x.state+(x.crown===flCrown?' cur':'');
  const t=document.createElement('b');t.textContent='Tán '+(x.crown+1);const s=document.createElement('span');
  s.textContent=x.crown===flCrown?'đang mở':(FL_STATE[x.state]||x.state)+(x.state!=='todo'?' · '+(100*x.cover).toLocaleString('vi-VN',{maximumFractionDigits:0})+'%':'');
  b.appendChild(t);b.appendChild(s);b.onclick=()=>flOpen(x.crown);box.appendChild(b);});}
// --- vẽ ---------------------------------------------------------------------
function flDrawCrop(im,alpha){if(!im)return;const [x0,y0,x1,y1]=im.box;ctx.save();ctx.globalAlpha=alpha;ctx.imageSmoothingEnabled=false;
 ctx.drawImage(im.bmp,sx(x0),sy(y0),(x1-x0)*scale,(y1-y0)*scale);ctx.restore();}
function flDraw(){if(wsActive!=='flower')return;
 const C={todo:'rgba(226,232,240,.75)',doing:'#fbbf24',done:'#2dd4bf'};
 ctx.save();
 flCrowns.forEach(c=>{if(c.crown===flCrown||(flCrown>=0&&!flLayers.others))return;
  polyPath(c.points);ctx.lineWidth=1.5;
  // Tán ngoài phần việc của mình: nét đứt mờ, bấm vào không mở.
  if(c.in_round===false){ctx.setLineDash([4,4]);ctx.strokeStyle='rgba(148,163,184,.45)';}else ctx.strokeStyle=C[c.state]||C.todo;
  ctx.stroke();ctx.setLineDash([]);});
 const c=flCur();
 if(c){// ngoài tán mờ đi để mắt chỉ còn một tán
  ctx.beginPath();ctx.rect(0,0,cv.width,cv.height);const p=c.points;
  for(let k=0;k<p.length;k+=2){k===0?ctx.moveTo(sx(p[k]),sy(p[k+1])):ctx.lineTo(sx(p[k]),sy(p[k+1]));}
  ctx.closePath();ctx.fillStyle='rgba(3,7,14,.6)';ctx.fill('evenodd');
  if(flLayers.mask)flDrawCrop(flImgs.mask,flOpacity);
  if(flLayers.human)flDrawCrop(flImgs.human,0.85);
  polyPath(c.points);ctx.lineWidth=2;ctx.strokeStyle='#2dd4bf';ctx.stroke();
  if(flStroke&&flStroke.pts.length){const t=FL_TOOLS[flStroke.tool];ctx.lineCap='round';ctx.lineJoin='round';
   ctx.strokeStyle=hexA(t.color,0.8);ctx.lineWidth=Math.max(1,2*flStroke.r*scale);ctx.beginPath();
   flStroke.pts.forEach((q,i)=>{i?ctx.lineTo(sx(q[0]),sy(q[1])):ctx.moveTo(sx(q[0]),sy(q[1]));});
   if(flStroke.pts.length===1)ctx.lineTo(sx(flStroke.pts[0][0])+0.01,sy(flStroke.pts[0][1]));ctx.stroke();}
  if(flMouse&&!flSpace&&!pan&&flTool!=='pan'){ctx.beginPath();ctx.arc(sx(flMouse[0]),sy(flMouse[1]),flBrush/2,0,7);
   ctx.lineWidth=1.5;ctx.strokeStyle=FL_TOOLS[flTool].color;ctx.stroke();}}
 ctx.restore();}
function flCursor(x,y){cv.style.cursor=(flSpace||flTool==='pan')?'grab':(flCrown>=0?'none':(flHit(x,y)>=0?'pointer':'grab'));}
wsRegister({id:'flower',
 resetImage(){flCrowns=[];flCrownsFor=null;flCrown=-1;flImgs={};flStroke=null;},
 onImage(){if(wsActive==='flower')flLoadCrowns();},
 draw(){flDraw();}});
tabRegister({id:'flower',inspTitle:'Tán đang gán',
 help:[['A · R','Chấp nhận · Sửa (vòng duyệt)'],['1 · 2 · 3 · E','Hoa · Không hoa · Bỏ qua · Tẩy'],['H','Kéo ảnh: chuột trái kéo để di chuyển, bấm vào tán để mở'],['[ · ]','Cọ nhỏ hơn · lớn hơn'],['T','Học & tô (vòng nét vẽ)'],['G','Bám siêu điểm ảnh (bộ vàng)'],['O','Ẩn / hiện mask hoa'],
  ['Enter','Xong tán này, sang tán kế'],['N','Tán chưa xong kế tiếp'],['Esc','Đóng tán đang mở'],['Ctrl Z','Bỏ thao tác cuối của tán'],['Space + kéo','Di chuyển ảnh'],['0','Vừa tán đang mở']],
 async enter(){if(!flRounds.length)await flLoadConfig();flLoadQueue();setFlTool(flTool);setFlBrush(flBrush);setFlOpacity(flOpacity);flRenderLayers();
  if(flCrownsFor!==current)await flLoadCrowns();else{flRender();draw();}
  if(flCrown<0)$('hint').textContent='Bấm vào một tán để bắt đầu gán hoa · N mở tán chưa xong kế tiếp.';
  if(!flTick)flTick=setInterval(flTimerTick,1000);},
 leave(){if(flCrown>=0)flClose();flMouse=null;flSpace=false;$('stTimer').textContent='';},
 key(e){if((e.ctrlKey||e.metaKey)&&(e.key==='z'||e.key==='Z')){e.preventDefault();flUndo();return true;}
  if(e.ctrlKey||e.metaKey||e.altKey)return false;
  const k=e.key;
  if(k===' '){e.preventDefault();flSpace=true;cv.style.cursor='grab';return true;}
  if(k==='1')setFlTool('hoa');else if(k==='2')setFlTool('khong');else if(k==='3')setFlTool('boqua');else if(k==='e'||k==='E')setFlTool('tay');else if(k==='h'||k==='H')setFlTool('pan');
  else if(k==='[')setFlBrush(flBrush/1.25);else if(k===']')setFlBrush(flBrush*1.25);
  else if(k==='t'||k==='T')flClassify();else if((k==='g'||k==='G')&&!$('flSnap').hidden)setFlSnap(!flSnap);else if(k==='o'||k==='O')toggleFlLayer('mask');
  else if((k==='a'||k==='A')&&flRoundObj()&&flRoundObj().kind==='review')flReview('accept');
  else if((k==='r'||k==='R')&&flRoundObj()&&flRoundObj().kind==='review')flReview('reject');
  else if(k==='Enter')flFinish();else if(k==='n'||k==='N')flNext();else if(k==='Escape')flClose();
  else if(k==='0'){if(flCrown>=0)flFitCrown();else resetView();}
  else return false;
  return true;},
 mouse(type,e){
  if(type==='down'){if(e.button!==0)return true;
   const [x,y]=toImg(e);
   const startPan=()=>{pan=[e.clientX,e.clientY,ox,oy];cv.style.cursor='grabbing';};
   if(flSpace||flTool==='pan'){flPanAt=[e.clientX,e.clientY,x,y];startPan();return true;}
   const k=flHit(x,y);
   if(flCrown<0){if(k>=0)flOpen(k);else startPan();return true;}
   // Bấm trúng một tán KHÁC thì chuyển sang tán đó; còn lại là vẽ (kể cả bắt
   // đầu nét từ ngoài mép tán: server cắt nét theo polygon).
   if(k>=0&&k!==flCrown&&!pip(flCur().points,x,y)){flOpen(k);return true;}
   flTouch();flStroke={tool:flTool,r:Math.max(0.5,(flBrush/2)/scale),snap:flSnap,pts:[[x,y]]};draw();return true;}
  if(type==='move'){const [x,y]=toImg(e);flMouse=[x,y];
   if(flStroke){const q=flStroke.pts[flStroke.pts.length-1];if(Math.hypot(x-q[0],y-q[1])*scale>=2)flStroke.pts.push([x,y]);}
   flCursor(x,y);draw();return true;}
  if(type==='up'){if(flStroke){const s=flStroke;flStroke=null;flSendStroke(s);}return true;}
  if(type==='leave'){flMouse=null;draw();return true;}
  if(type==='panEnd'){const at=flPanAt;flPanAt=null;
   // Kéo ảnh mà chỉ bấm (không kéo) lên một tán khác: mở tán đó.
   if(at&&flTool==='pan'&&Math.abs(e.clientX-at[0])<3&&Math.abs(e.clientY-at[1])<3){const k=flHit(at[2],at[3]);if(k>=0&&k!==flCrown)flOpen(k);}
   if(flMouse)flCursor(flMouse[0],flMouse[1]);return true;}
  return true;}});
// --- vòng đã dựng: tiến độ, hàng đợi, đồng thuận -------------------------------
function flGo(it){if(it.name===current){flOpen(it.crown);return;}flPending={name:it.name,crown:it.crown};selectImg(it.name);}
async function flLoadQueue(){const r=flRoundObj(),tok=++flQTok;
 if(!r||(r.pick&&r.pick.mode==='manual')||!r.built){flQueue=null;flRenderQueue();return;}
 let d=null;try{d=await(await fetch('/api/fl/queue?round='+encodeURIComponent(flRound)+'&who='+encodeURIComponent(annot),{cache:'no-store'})).json();}catch(e){d=null;}
 if(tok!==flQTok)return;flQueue=(d&&d.ok)?d:null;flRenderQueue();}
function flRenderQueue(){const r=flRoundObj(),manualPick=!!(r&&r.pick&&r.pick.mode==='manual'),built=!!(r&&r.built);
 $('flBuildBox').hidden=!(r&&!manualPick&&!built);$('flCloseBox').hidden=!(r&&r.kind==='review'&&built);
 $('flSnap').hidden=!(r&&r.kind==='manual');if($('flSnap').hidden&&flSnap)setFlSnap(false);
 $('flAgreeBox').hidden=!(r&&built&&(r.double||r.kind==='manual'));$('flAgreeAB').hidden=!(r&&r.double);
 const pg=$('flProg');pg.innerHTML='';const q=$('flQueue');q.innerHTML='';
 if(!flQueue){$('flQCount').textContent=manualPick?'chọn tay':'—';
  if(manualPick){const s=document.createElement('span');s.className='muted';s.textContent='Vòng chọn tay: bấm vào tán trên ảnh để gán.';q.appendChild(s);}
  return;}
 Object.entries(flQueue.progress).forEach(([f,p])=>{const row=document.createElement('div');row.className='row';
  const a=document.createElement('span');a.textContent=f;const b=document.createElement('b');const i=document.createElement('i');
  i.style.width=(p.total?100*p.done/p.total:0)+'%';b.appendChild(i);const c=document.createElement('span');c.textContent=p.done+'/'+p.total;
  row.appendChild(a);row.appendChild(b);row.appendChild(c);pg.appendChild(row);});
 const info=document.createElement('div');info.className='muted';
 info.textContent=(flQueue.second?'Người '+annot+': phần gán đôi · ':'')+flQueue.done+'/'+flQueue.total+(flQueue.kind==='review'?' tán đã duyệt':' tán xong')+
  (flQueue.median_seconds!=null?' · trung vị '+Math.round(flQueue.median_seconds)+' s/tán':'')+
  (flQueue.reject_rate!=null?' · phải sửa '+Math.round(100*flQueue.reject_rate)+'%':'');pg.appendChild(info);
 const left=flQueue.items.filter(x=>!FL_DONE.includes(x.state));$('flQCount').textContent=left.length+' còn';
 left.slice(0,40).forEach(x=>{const b=document.createElement('button');
  b.className=(x.state==='doing'?'doing':'')+(x.name===current&&x.crown===flCrown?' cur':'');
  const i=document.createElement('i');const s=document.createElement('span');s.textContent=x.field+' · '+frameLabel(x.name)+' · tán '+(x.crown+1);
  b.title=x.name+' · tán '+(x.crown+1);b.appendChild(i);b.appendChild(s);
  if(x.reason){const em=document.createElement('em');em.textContent=x.reason==='uncertain'?('kém chắc'+(x.unc!=null?' '+x.unc.toFixed(2).replace('.',','):'')):'ngẫu nhiên';b.appendChild(em);}
  b.onclick=()=>flGo(x);q.appendChild(b);});}
function setFlSnap(on){flSnap=on;$('flSnap').setAttribute('aria-pressed',on);}
async function flBuild(){try{const j=await flApi('/api/fl/round/build',{round:flRound});
  $('flBuildInfo').textContent='✔ '+j.n_items+' tán · '+Object.entries(j.by_field).map(([f,v])=>f+' '+v.crowns).join(' · ');
  await flLoadConfig();await flLoadQueue();await flLoadCrowns();}catch(e){flErr(e);}}
async function flAgree(against){const box=$('flAgree');box.textContent='⏳ đang tính…';
 let d=null;try{d=await(await fetch('/api/fl/agreement?round='+encodeURIComponent(flRound)+(against?'&against='+encodeURIComponent(against):''),{cache:'no-store'})).json();}catch(e){d={ok:false,error:String(e)};}
 if(!d.ok){box.textContent='⚠ '+d.error;return;}
 const f=v=>v==null?'—':v.toFixed(3).replace('.',',');
 const rows=Object.entries(d.by_field).map(([k,v])=>'<tr><td>'+k+'</td><td>'+v.crowns+'</td><td>'+f(v.iou)+'</td><td>'+f(v.f1)+'</td></tr>').join('');
 box.innerHTML='<table><tr><th>Ruộng</th><th>Tán</th><th>IoU hoa</th><th>F1 hoa</th></tr>'+rows+
  '<tr><td><b>Tất cả</b></td><td>'+d.all.crowns+'</td><td><b>'+f(d.all.iou)+'</b></td><td><b>'+f(d.all.f1)+'</b></td></tr></table>'+
  '<div class="muted">'+(against?'Bộ phân loại vòng '+against+' so với mask đã chốt của vòng này':'Người gán chính so với người thứ hai')+' · bỏ pixel Bỏ qua ở bất kỳ bên nào.</div>';}
// Đổi người gán: phần việc và lớp nét đổi theo người.
function flOnAnnot(){if(wsActive!=='flower')return;flCrown=-1;flImgs={};
 $('hint').textContent='Người gán '+annot+' · phần việc và nét vẽ đổi theo người.';flLoadQueue();flLoadCrowns();}
// --- vòng duyệt ----------------------------------------------------------------
async function flReview(action){const r=flRoundObj();if(flCrown<0||flBusy||!r||r.kind!=='review')return;flTouch();
 try{const j=await flApi('/api/fl/review',flBody({action}));const c=flCur();if(c)c.state=j.state;flApply(j);
  if(flQueue){const q=flQueue.items.find(x=>x.name===current&&x.crown===flCrown);if(q)q.state=j.state;}
  if(action==='accept'){flLoadQueue();flNext();}
  else $('hint').textContent='Đang sửa: vẽ nét đúng nhãn lên chỗ model sai (mask vàng là dự đoán), rồi Xong (Enter).';}
 catch(e){flErr(e);}}
function flCloseRound(){askConfirm({title:'Chốt vòng '+flRound+'?',ok:'Chốt vòng',
  text:'Mọi tán của vòng chưa được nhận hay sửa sẽ thành "tự nhận": lấy nguyên dự đoán của model, chưa ai xem. Tán đang sửa dở giữ nguyên.'},
 async()=>{try{const j=await flApi('/api/fl/round/close',{round:flRound});
  $('hint').textContent='✔ Đã chốt vòng: '+j.auto+' tán tự nhận'+(j.missing_prediction?(' · ⚠ '+j.missing_prediction+' tán chưa có dự đoán'):'');
  flLoadQueue();flLoadCrowns();}catch(e){flErr(e);}});}
