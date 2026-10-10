// ws_crown.js — sửa polygon tán: SAM, vẽ tay, nắn đỉnh, danh sách vùng, lưu.
const HANDLE_PX=16,INFLUENCE_PX=60;
// INFLUENCE_PX là bán kính bút, CỐ ĐỊNH theo px màn hình: kéo xa hay gần thì
// vẫn đúng ngần ấy đoạn biên nhúc nhích, nên nắn được đúng chỗ mình muốn.
// Muốn ảnh hưởng rộng hơn thì zoom ra, hẹp hơn thì zoom vào — mức zoom chính
// là núm chỉnh cỡ bút (60px màn hình ≈ 23px ảnh ở 1200%, ≈ 270px ở cỡ vừa khung).
// BRUSH_STEP_PX = khoảng cách đỉnh chèn thêm dưới bút. SIMPLIFY_PX/WELD_PX tính
// bằng px ẢNH (KHÔNG theo zoom) — mấu chốt giữ số đỉnh không phình theo zoom.
const BRUSH_STEP_PX=8,SIMPLIFY_PX=0.6,WELD_PX=0.3;
// Bán kính bắt CẠNH (px màn hình). Nhỏ hơn bán kính bắt đỉnh (11) để đỉnh
// thật luôn thắng khi hai thứ nằm chồng nhau.
const EDGE_PX=8;
let inst=[],store={},hist=[],saveTimer=null,rid=1,dirty=false;
// Ảnh "đã xem, không có cây": nhãn rỗng được LƯU (khác với chưa gán) để ảnh
// nền đi vào bộ xuất. emptyMark = ảnh hiện tại; emptyStore = theo ảnh, cùng
// vòng đời với store; emptySet = server báo ảnh nào đang có nhãn rỗng.
let emptyMark=false,emptyStore={},emptySet=new Set();
let lastEdit='select',mode='sam-click',sel=-1,dragPt=-1,dragVi=-1,dragStart=null,origPoly=null,softD=null,dragSoft=true,hovPt=-1,ghost=null,draft=[],boxStart=null,mouse=null,pan=null;
const HINT={'sam-click':'SAM Click · Bấm vào 1 cây để tạo vùng mới (lớp đang chọn).','sam-box':'SAM Box · Kéo 1 khung quanh cây để tạo vùng chính xác hơn.','draw':'Vẽ tay · Click các điểm quanh tán; double-click để đóng vùng; Esc để huỷ.','select':'Nắn · Kéo điểm trắng để nắn viền (Shift = 1 điểm) · rê lên cạnh để hiện điểm mờ, bấm là thêm điểm ngay đó · chuột phải xoá điểm · Delete xoá vùng · Ctrl+Z hoàn tác.',
 'look':'👁 Xem · nhãn tán đã chốt nên không sửa được · kéo để di chuyển, lăn để phóng to · bấm ✎ Sửa khi thật sự cần.',
 'view':'👁 Xem dự đoán — KHOÁ SỬA · kéo để di chuyển, lăn để phóng to · rê lên một vùng để xem điểm và IoU · bỏ tick "Hiện dự đoán" để sửa lại.'};
const BTN={'sam-click':'m-click','sam-box':'m-box','draw':'m-draw','select':'m-sel'};
// 'view' = chế độ CHỈ XEM. Khoá bằng một chế độ riêng chứ không rải cờ khắp
// nơi: bốn nhánh sửa trong handler chuột đều bắt đầu bằng `mode==='...'` nên
// không nhánh nào khớp nữa, tức không có đường nào lọt. Các nút trong danh
// sách vùng và phím tắt đi đường riêng nên vẫn hỏi `predLocked()`.
function setMode(m){if(predOn&&m!=='view')return;
 mode=m;if(m!=='view')lastEdit=m;draft=[];boxStart=null;hovPt=-1;ghost=null;dragPt=-1;dragVi=-1;origPoly=null;softD=null;
 document.querySelectorAll('#editSeg button').forEach(b=>b.classList.remove('on'));
 if(BTN[m])$(BTN[m]).classList.add('on');
 cv.style.cursor=(m==='view')?'grab':'crosshair';$('hint').textContent=((m==='view'&&!predOn)?HINT.look:HINT[m])||'';syncVE();draw();}
// Thẻ Tán có hai chế độ: Xem (mặc định, không gì sửa được nhãn đã chốt) và Sửa
// (bốn công cụ cũ). Lúc xem dự đoán thì không bật Sửa được.
function syncVE(){const v=(mode==='view');$('veView').classList.toggle('on',v);$('veEdit').classList.toggle('on',!v);
 $('veEdit').disabled=predOn;$('editTools').hidden=v;renderRegions();}
function regionArea(r){const p=r.poly,n=p.length/2;let a=0;for(let i=0;i<n;i++){const j=(i+1)%n;a+=p[i*2]*p[j*2+1]-p[j*2]*p[i*2+1];}return Math.abs(a)/2;}
// Dựng lại cả danh sách thì scrollTop về 0: ảnh 30-48 vùng, bấm 👁/⚡/đổi lớp ở
// hàng dưới là hàng đó bay khỏi tầm nhìn, trông như vùng đã mất. Giữ lại vị
// trí cuộn; riêng khi VÙNG ĐƯỢC CHỌN đổi (bấm trên canvas) thì cuộn tới nó.
let lastSelRow=-1;
function renderRegions(){const box=$('regions');const st=box.scrollTop;box.innerHTML='';$('rcount').textContent=inst.length;
 inst.forEach((r,i)=>{const c=flowerOn?(clsById(r.cls)||{name:'?',color:'#888'}):{name:'',color:CROWN_COLOR};
  const row=document.createElement('div');row.className='region'+(i===sel?' sel':'');
  const head=document.createElement('div');head.className='rhead';
  const dot=document.createElement('span');dot.className='dot';dot.style.background=c.color;
  const idb=document.createElement('b');idb.textContent='#'+r.id;
  const cn=document.createElement('span');cn.className='rcls';
  // Tắt đếm hoa: hàng chỉ còn số và diện tích tán, không còn mức/%/⚡.
  if(flowerOn){cn.textContent=c.name+(r.manual?' ✋':'');
   cn.title=r.manual?'Mức do bạn gán tay — máy sẽ không ghi đè. Bấm ⚡ để phân loại lại tự động.':'Mức do '+methodName()+' tự phân loại';}
  else cn.textContent=Math.round(regionArea(r)).toLocaleString('vi-VN')+' px';
  const pc=document.createElement('span');pc.className='rpct';pc.textContent=flowerOn?(r.pct==null?'—':'🌸 '+r.pct.toFixed(1)+'%'):'';
  const ps=r.pcts||{},od=r.od;
  pc.title='Tỉ lệ pixel hoa ('+methodName()+')\nHSV '+pctTxt(ps.hsv)+' · Otsu '+pctTxt(ps.otsu)+(od?' ['+(od.channel||'')+' T='+(od.thr==null?'—':Math.round(od.thr))+(od.thr1!=null?' T1='+Math.round(od.thr1):'')+' η='+(od.sep==null?'—':(+od.sep).toFixed(2))+(od.class_s!=null?' S lớp='+Math.round(od.class_s):'')+' '+(od.n_kept==null?'':od.n_kept+'/'+od.n_blobs+' blob')+(od.rejected?' '+od.rejected:'')+']':'');
  head.title='#'+r.id+' · '+(r.poly.length/2)+' đỉnh';
  head.appendChild(dot);head.appendChild(idb);head.appendChild(cn);head.appendChild(pc);
  // Chọn từ danh sách thì vùng phải hiện: vùng được chọn luôn nhìn thấy được.
  // Ở chế độ xem dự đoán, bấm một hàng chỉ để HIỆN lại vùng đó: `tidySel` dọn
  // polygon rồi `persist()` lưu xuống đĩa, tức nó là một thao tác sửa.
  head.onclick=()=>{if(!r.vis){if(!predLocked())pushHist();r.vis=true;}
   if(predLocked()){renderRegions();draw();return;}
   sel=i;setMode('select');tidySel(i);renderRegions();draw();};
  const act=document.createElement('div');act.className='ract';
  const selc=document.createElement('select');selc.title='Gán mức mật độ thủ công';classes.forEach(cc=>{const o=document.createElement('option');o.value=cc.id;o.textContent=cc.name;if(cc.id===r.cls)o.selected=true;selc.appendChild(o);});
  selc.onchange=()=>{pushHist();r.cls=parseInt(selc.value);r.manual=true;persist();draw();renderRegions();};
  const bF=document.createElement('button');bF.textContent='⚡';bF.title='Đếm hoa & tự phân loại vùng này';bF.onclick=()=>{pushHist();computeFlower([i],true);};
  // Ẩn/hiện đi vào lịch sử: trước đây Ctrl+Z ngay sau khi ẩn hoàn tác thao tác
  // TRƯỚC ĐÓ (thường là chính lần SAM click tạo ra vùng) -> vùng mất thật.
  // Ẩn vùng đang chọn thì bỏ chọn luôn, để phím Delete và chuột trên canvas
  // không còn nhắm vào thứ không nhìn thấy.
  const bV=document.createElement('button');bV.textContent=r.vis?'👁':'🚫';bV.title=r.vis?'Ẩn vùng này (chỉ trên màn hình, vẫn lưu)':'Hiện lại vùng này';
  bV.onclick=()=>{pushHist();r.vis=!r.vis;if(!r.vis&&sel===i){sel=-1;dragPt=-1;hovPt=-1;ghost=null;}draw();renderRegions();};
  const bD=document.createElement('button');bD.className='del';bD.textContent='🗑';bD.title='Xoá vùng';
  bD.onclick=()=>{pushHist();inst.splice(i,1);if(sel===i)sel=-1;else if(sel>i)sel--;persist();draw();renderRegions();};
  // 👁 chỉ đổi trạng thái XEM (`vis` không đi vào payload lưu) nên vẫn mở ở
  // chế độ xem dự đoán — ẩn một nhãn đi để nhìn rõ đường dự đoán bên dưới là
  // đúng việc người ta cần lúc so sánh. Ba nút còn lại đều ghi xuống đĩa.
  [selc,bF,bD].forEach(el=>{if(predLocked()){el.disabled=true;el.style.opacity=.4;
   el.style.cursor='not-allowed';el.title=predOn?'Đang ở chế độ xem dự đoán — tắt để sửa lại':'Đang ở chế độ Xem — bấm ✎ Sửa để chỉnh';}});
  if(flowerOn){act.appendChild(selc);act.appendChild(bF);}
  act.appendChild(bV);act.appendChild(bD);
  row.appendChild(head);row.appendChild(act);box.appendChild(row);});
 box.scrollTop=st;
 if(sel>=0&&sel!==lastSelRow){const el=box.children[sel];if(el&&el.scrollIntoView)el.scrollIntoView({block:'nearest'});}
 lastSelRow=sel;}
function cloneR(r){return {poly:r.poly.slice(),cls:r.cls,vis:r.vis,id:r.id,pct:(r.pct==null?null:r.pct),manual:!!r.manual,pcts:r.pcts?{hsv:r.pcts.hsv,otsu:r.pcts.otsu}:null,od:r.od||null};}
function payload(){return JSON.stringify(Object.assign({name:current,width:img.width||0,height:img.height||0,polygons:inst.map(r=>r.poly),classes:inst.map(r=>r.cls),manual:inst.map(r=>!!r.manual),categories:classes,empty:(emptyMark&&!inst.length),flower:flowerOn},flowerParams()));}
async function saveNow(){if(!current||!ready)return;dirty=false;
 const who=current;svst('','⏳ đang lưu…');
 try{const r=await(await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:payload()})).json();
  if(current!==who)return;
  if(r.warning)svst('err','⚠ '+r.warning,r.warning);
  else svst('ok','✔ đã lưu '+hhmm(),'Nhãn lưu tại: '+(saveDirStart||'(mặc định)'));}
 catch(e){if(current===who)svst('err','⚠ chưa lưu được','Không gửi được về server: '+e);}}
function scheduleSave(){if(!current)return;dirty=true;clearTimeout(saveTimer);saveTimer=setTimeout(saveNow,700);}
function flush(){clearTimeout(saveTimer);if(dirty)saveNow();}
function beacon(){if(!current||!ready||!dirty)return;try{navigator.sendBeacon('/api/save',new Blob([payload()],{type:'application/json'}));dirty=false;}catch(e){}}
function stash(){if(current){store[current]=inst.map(cloneR);emptyStore[current]=emptyMark;}}
// Vẽ được một vùng là ảnh hết "không có cây"; xoá hết vùng thì về chưa gán
// chứ không tự thành không-có-cây — cờ đó phải do người bấm.
function persist(){if(inst.length)emptyMark=false;stash();scheduleSave();
 if(current){if(inst.length||emptyMark)labeledSet.add(current);else labeledSet.delete(current);
  if(emptyMark&&!inst.length)emptySet.add(current);else emptySet.delete(current);}
 renderStats();updateThumbActive();renderClasses();syncEmptyBtn();scheduleMask();}
function countText(){return inst.length+' Vùng'+((emptyMark&&!inst.length)?' · không có cây':'');}
function syncEmptyBtn(){const b=$('noTree');if(!b)return;const on=emptyMark&&!inst.length;
 b.classList.toggle('on',on);b.disabled=!current||!ready||inst.length>0;
 b.title=inst.length?'Ảnh đang có '+inst.length+' vùng — xoá hết trước nếu muốn đánh dấu không có cây':
  (on?'Đang đánh dấu không có cây. Bấm lại để bỏ (ảnh trở về chưa gán).':'Ảnh này đã xem và không có cây cà phê. Được lưu như một nhãn rỗng và đi vào bộ xuất như ảnh nền.');}
function refresh(){$('count').textContent=countText();persist();renderRegions();draw();}
function pushHist(){hist.push(JSON.stringify(inst));if(hist.length>60)hist.shift();}
function undo(){if(!hist.length)return;inst=JSON.parse(hist.pop());sel=-1;dragPt=-1;hovPt=-1;refresh();}
function addRegion(poly){inst.push({poly:poly,cls:activeCls,vis:true,id:rid++,pct:null,manual:false,pcts:null,od:null});return inst.length-1;}
function ringArea(p){let a=0;const n=p.length/2;
 for(let i=0;i<n;i++){const j=(i+1)%n;a+=p[i*2]*p[j*2+1]-p[j*2]*p[i*2+1];}return a/2;}
// Chèn thêm đỉnh CHỈ trong cung nằm dưới bút, quanh đỉnh vi (rad tính bằng px
// ẢNH). Bản cũ densify CẢ polygon theo zoom hiện tại và không bao giờ xoá, nên
// số đỉnh chỉ có tăng và tích luỹ theo mức zoom sâu nhất từng dùng — đo được
// median 103 đỉnh/tán (SAM chỉ trả 20-40) và có tán chạm 1480 đỉnh.
// Trả {poly,vi} vì việc chèn làm dịch chỉ số của chính đỉnh đang kéo.
function densifyArc(p,vi,rad,maxLen){const n=p.length/2;
 if(n<3||!(maxLen>0))return {poly:p.slice(),vi:vi};
 const mark=new Uint8Array(n);
 for(let dir=0;dir<2;dir++){let acc=0,cur=vi;
  for(let s=0;s<n;s++){const nxt=dir?(cur-1+n)%n:(cur+1)%n;
   mark[dir?nxt:cur]=1;
   acc+=Math.hypot(p[nxt*2]-p[cur*2],p[nxt*2+1]-p[cur*2+1]);
   if(acc>rad)break;
   cur=nxt;}}
 const out=[];let nvi=vi;
 for(let i=0;i<n;i++){const a=i*2,b=((i+1)%n)*2;
  if(i===vi)nvi=out.length/2;
  out.push(p[a],p[a+1]);
  if(!mark[i])continue;
  const ax=p[a],ay=p[a+1],bx=p[b],by=p[b+1];
  const m=Math.max(0,Math.ceil(Math.hypot(bx-ax,by-ay)/maxLen)-1);
  for(let j=1;j<=m;j++){const t=j/(m+1);out.push(ax+(bx-ax)*t,ay+(by-ay)*t);}}
 return {poly:out,vi:nvi};}
// Gộp các đỉnh sát nhau hơn minD (px ảnh). Kéo tràn ra ngoài khung rồi bị cX/cY
// kẹp vào mép sinh ra cả loạt đỉnh trùng nhau, có cạnh dài đúng 0,00px.
function weldPoly(p,minD){const n=p.length/2;if(n<4)return p.slice();
 const out=[],m2=minD*minD;
 for(let i=0;i<n;i++){const x=p[i*2],y=p[i*2+1];
  if(out.length){const dx=x-out[out.length-2],dy=y-out[out.length-1];
   if(dx*dx+dy*dy<m2)continue;}
  out.push(x,y);}
 if(out.length>=6){const dx=out[0]-out[out.length-2],dy=out[1]-out[out.length-1];
  if(dx*dx+dy*dy<m2)out.length-=2;}
 return out.length>=6?out:p.slice();}
// Ramer-Douglas-Peucker: bỏ đỉnh lệch dưới eps (px ẢNH) so với dây cung — tức
// đỉnh không mang thông tin hình dạng nào. Chốt 2 đỉnh xa nhau nhất để cắt vòng
// kín thành hai cung, tránh việc đỉnh số 0 luôn được giữ làm lệch kết quả.
function rdpPoly(p,eps){const n=p.length/2;if(n<4)return p.slice();
 let b=0,bd=-1;
 for(let i=1;i<n;i++){const d=Math.hypot(p[i*2]-p[0],p[i*2+1]-p[1]);
  if(d>bd){bd=d;b=i;}}
 if(b<1)return p.slice();
 const keep=new Uint8Array(n);keep[0]=keep[b]=1;
 const st=[[0,b],[b,n]];
 while(st.length){const seg=st.pop(),i=seg[0],j=seg[1];
  if(j<=i+1)continue;
  const ax=p[(i%n)*2],ay=p[(i%n)*2+1];
  const ex=p[(j%n)*2]-ax,ey=p[(j%n)*2+1]-ay,L2=ex*ex+ey*ey;
  let m=-1,md=eps;
  for(let k=i+1;k<j;k++){const kk=k%n,px=p[kk*2],py=p[kk*2+1];
   let t=L2>0?((px-ax)*ex+(py-ay)*ey)/L2:0;t=t<0?0:(t>1?1:t);
   const d=Math.hypot(px-ax-t*ex,py-ay-t*ey);
   if(d>md){md=d;m=k;}}
  if(m<0)continue;
  keep[m%n]=1;st.push([i,m]);st.push([m,j]);}
 const out=[];
 for(let i=0;i<n;i++)if(keep[i])out.push(p[i*2],p[i*2+1]);
 return out.length>=6?out:p.slice();}
// Giao của hai đoạn thẳng, chỉ tính khi cắt THỰC SỰ ở phần trong (bỏ đầu mút).
function segX(ax,ay,bx,by,cx0,cy0,dx0,dy0){
 const r1=bx-ax,r2=by-ay,s1=dx0-cx0,s2=dy0-cy0,den=r1*s2-r2*s1;
 if(Math.abs(den)<1e-12)return null;
 const t=((cx0-ax)*s2-(cy0-ay)*s1)/den,u=((cx0-ax)*r2-(cy0-ay)*r1)/den;
 if(t<=0||t>=1||u<=0||u>=1)return null;
 return [ax+t*r1,ay+t*r2];}
// Gỡ nút thắt: chỗ biên tự cắt chia polygon thành hai vòng, giữ vòng LỚN hơn.
// Đây mới là thứ chữa "móc chéo" — kẹp theo khoảng cách không gian thì vô ích
// vì khoảng cách thẳng luôn <= khoảng cách dọc biên, cửa chặn không bao giờ mở.
function untanglePoly(p,maxIter){let poly=p.slice();
 for(let it=0;it<(maxIter||8);it++){const n=poly.length/2;
  if(n<4)break;
  let hit=null;
  for(let i=0;i<n&&!hit;i++){const a=i*2,b=((i+1)%n)*2;
   for(let j=i+2;j<n;j++){
    if(i===0&&j===n-1)continue;
    const c=j*2,d=((j+1)%n)*2;
    const X=segX(poly[a],poly[a+1],poly[b],poly[b+1],
                 poly[c],poly[c+1],poly[d],poly[d+1]);
    if(X){hit=[i,j,X];break;}}}
  if(!hit)break;
  const i=hit[0],j=hit[1],X=hit[2];
  const A=[X[0],X[1]];
  for(let k=i+1;k<=j;k++)A.push(poly[k*2],poly[k*2+1]);
  const B=[X[0],X[1]];
  for(let k=j+1;k<n+i+1;k++){const kk=k%n;B.push(poly[kk*2],poly[kk*2+1]);}
  const keepA=Math.abs(ringArea(A))>=Math.abs(ringArea(B));
  const nx=keepA?A:B;
  if(nx.length<6)break;
  poly=nx;}
 return poly.length>=6?poly:p.slice();}
function samePts(a,b){if(a.length!==b.length)return false;
 for(let i=0;i<a.length;i++)if(a[i]!==b[i])return false;
 return true;}
// Dọn sau MỖI nhát nắn: gộp đỉnh trùng -> bỏ đỉnh thừa -> gỡ nút tự cắt.
// Nhờ bước này số đỉnh CÂN BẰNG lại (chèn khi cần nắn, bỏ khi hết tác dụng)
// thay vì chỉ tăng một chiều theo mức zoom. Ngưỡng tính bằng px ảnh nên kết quả
// giống hệt nhau dù bạn nắn ở 100% hay 1200%.
// LẶP tới điểm bất động: untangle đổi thứ tự đỉnh nên RDP ở vòng sau còn bỏ được
// thêm vài đỉnh (đo trên tán thật: 383 -> 265 -> 264 -> đứng yên). Chốt vòng lặp
// ngay đây để cleanPoly IDEMPOTENT tuyệt đối — điều kiện bắt buộc mới dám dọn
// lúc CHỌN polygon, xem commit a743905.
function cleanPoly(p){let q=p.slice();
 for(let it=0;it<4;it++){
  const r=untanglePoly(rdpPoly(weldPoly(q,WELD_PX),SIMPLIFY_PX));
  if(samePts(r,q))return r;
  q=r;}
 return q;}
// Dọn polygon NGAY KHI CHỌN, để các tán đã lỡ phình/móc chéo từ trước tự gọn lại
// khi bạn bấm vào, không phải nắn tay từng cái.
// Commit a743905 đã gỡ đúng hành vi này vì resamplePoly lấy mẫu theo
// HANDLE_PX/scale: cùng một tán cho ra 26..1500 đỉnh tuỳ zoom và cỡ màn hình, và
// mỗi lần chọn lại bào thêm một ít. cleanPoly không có cả hai tật đó — ngưỡng
// tính bằng px ẢNH nên mọi màn hình ra cùng một kết quả, và đã bất động nên
// chọn 100 lần đúng bằng chọn 1 lần.
function tidySel(i){const r=inst[i];if(!r)return;
 const before=r.poly.length/2,np=cleanPoly(r.poly);
 if(samePts(np,r.poly))return;     // đã sạch -> không đụng lịch sử, không lưu lại
 pushHist();r.poly=np;
 $('hint').textContent='🧹 Đã dọn #'+r.id+': '+before+' → '+(np.length/2)+
  ' đỉnh · Ctrl+Z để hoàn tác';
 persist();}
// Tập đỉnh được VẼ làm tay cầm: cách nhau >= gap px trên màn hình. Chỉ ảnh hưởng
// hiển thị, dữ liệu polygon không đổi.
function handleIdx(p,gap){const n=p.length/2,out=[];let lx=0,ly=0;
 for(let i=0;i<n;i++){const x=p[i*2],y=p[i*2+1];
  if(!out.length||Math.hypot(x-lx,y-ly)*scale>=gap){out.push(i);lx=x;ly=y;}}
 if(out.length>2){const f=out[0]*2,l=out[out.length-1]*2;
  if(Math.hypot(p[f]-p[l],p[f+1]-p[l+1])*scale<gap*0.6)out.pop();}
 return out;}
// Khoảng cách dọc theo ĐƯỜNG BIÊN (px màn hình) từ đỉnh vi tới các đỉnh khác,
// đi cả hai chiều, dừng khi vượt limit. Nhờ đó bút nắn có bán kính thật theo
// màn hình, không phụ thuộc các đỉnh thưa hay dày.
function arcDistPx(p,vi,limit){const n=p.length/2,d=new Float64Array(n).fill(Infinity);d[vi]=0;
 for(let dir=0;dir<2;dir++){let acc=0,cur=vi;
  for(let s=1;s<n;s++){const nxt=dir?(cur-1+n)%n:(cur+1)%n;
   acc+=Math.hypot(p[nxt*2]-p[cur*2],p[nxt*2+1]-p[cur*2+1])*scale;
   if(acc>limit)break;
   if(acc<d[nxt])d[nxt]=acc;
   cur=nxt;}}
 return d;}
async function samCall(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(j.error){alert('SAM lỗi: '+j.error);return;}const ni=[];(j.polygons||[]).forEach(p=>ni.push(addRegion(p)));if(ni.length)sel=inst.length-1;refresh();if(ni.length)computeFlower(ni);}
// Chỉ bắt vào các tay cầm ĐANG HIỆN, để bấm trúng cái mình nhìn thấy.
function hitVertex(p,x,y){let best=-1,bd=11;
 for(const i of handleIdx(p,HANDLE_PX)){const d=Math.hypot(p[i*2]-x,p[i*2+1]-y)*scale;
  if(d<bd){bd=d;best=i*2;}}
 return best;}
// Điểm trên BIÊN gần con trỏ nhất, trong bán kính EDGE_PX px màn hình.
// Trả {i,x,y}: i = đỉnh đầu của cạnh, (x,y) = điểm chiếu (toạ độ ảnh).
// Sau khi dọn đỉnh thì biên thưa hẳn ra, không còn chỗ để nhắm; đỉnh giả cho
// phép nắn ở BẤT KỲ đâu trên cạnh mà không phải chèn sẵn đỉnh thừa khắp nơi.
function hitEdge(p,mx,my){const n=p.length/2;let best=null,bd=EDGE_PX;
 for(let i=0;i<n;i++){const a=i*2,b=((i+1)%n)*2;
  const ax=p[a],ay=p[a+1],ex=p[b]-ax,ey=p[b+1]-ay,L2=ex*ex+ey*ey;
  let t=L2>0?((mx-ax)*ex+(my-ay)*ey)/L2:0;t=t<0?0:(t>1?1:t);
  const qx=ax+ex*t,qy=ay+ey*t,d=Math.hypot(qx-mx,qy-my)*scale;
  if(d<bd){bd=d;best={i:i,x:qx,y:qy};}}
 return best;}
// Bắt đầu kéo đỉnh vi. Tách riêng vì giờ có hai lối vào: bấm trúng đỉnh có
// sẵn, và bấm vào đỉnh giả trên cạnh.
function beginDrag(vi,x,y,shift){
 // Chỉ chèn đỉnh và chỉ đo khoảng cách trong đúng tầm bút — ngoài tầm đó
 // trọng số bằng 0 nên tính thêm cũng vứt đi.
 const dz=densifyArc(inst[sel].poly,vi,INFLUENCE_PX/scale,
                     Math.max(BRUSH_STEP_PX/scale,SIMPLIFY_PX));
 inst[sel].poly=dz.poly;dragVi=dz.vi;dragPt=dragVi*2;
 dragStart=[x,y];origPoly=inst[sel].poly.slice();dragSoft=!shift;
 softD=arcDistPx(origPoly,dragVi,INFLUENCE_PX);
 ghost=null;hovPt=dragPt;cv.style.cursor='grabbing';}
function hitPolygon(x,y){for(let i=inst.length-1;i>=0;i--){if(!inst[i].vis)continue;if(pip(inst[i].poly,x,y))return i;}return -1;}
wsRegister({id:'crown',
 beforeImage(){if(current&&ready&&dirty){flush();}if(current&&ready){store[current]=inst.map(cloneR);}},
 resetImage(){dirty=false;draft=[];sel=-1;dragPt=-1;hovPt=-1;ghost=null;hist=[];inst=[];emptyMark=false;},
 async loadImage(t,tok){
 // Ẩn/hiện là trạng thái xem của LẦN đang mở, không theo ảnh: quay lại ảnh cũ
 // mọi vùng đều hiện, khỏi phải nhớ đã ẩn gì ở đâu.
 if(store[t]){inst=store[t].map(cloneR);inst.forEach(r=>{r.vis=true;});emptyMark=!!emptyStore[t];return;}
 let d=null;try{d=await(await fetch('/api/load/'+enc(t)+'?t='+Date.now(),{cache:'no-store'})).json();}catch(e){d=null;}
  if(tok!==selToken)return;
  if(d){mergeCats(d.categories);inst=(d.polygons||[]).map((pp,k)=>{const h=(d.hsv_ratios&&d.hsv_ratios[k]!=null)?d.hsv_ratios[k]:((d.ratios&&d.ratios[k]!=null)?d.ratios[k]:null);
    const o=(d.otsu_ratios&&d.otsu_ratios[k]!=null)?d.otsu_ratios[k]:null;const pcts={hsv:h,otsu:o};
    // Mở lại ảnh: hiện % của phương pháp đang chọn; nhãn giữ như đã lưu.
    return {poly:pp.slice(),cls:pickCls(d.classes&&d.classes[k]),vis:true,id:rid++,pct:pcts[fmethod],manual:!!(d.manual&&d.manual[k]),pcts:pcts,od:(d.otsu_info&&d.otsu_info[k])||null};});}else inst=[];
  emptyMark=!!(d&&d.empty&&!inst.length);
  store[t]=inst.map(cloneR);emptyStore[t]=emptyMark;},
 imageShown(){renderRegions();},
 onImage(){$('count').textContent=countText();syncEmptyBtn();renderRegions();},
 draw(){
 // Ở chế độ xem dự đoán, nhãn lùi về làm NỀN: nền nhạt hơn và viền mảnh hơn,
 // để lớp dự đoán vẽ đè lên còn đọc được. Bỏ hẳn nhãn đi thì mất vế so sánh,
 // nên đó là một ô tick riêng chứ không phải mặc định.
 if(!(predOn&&!predShowGt))inst.forEach((r,i)=>{if(!r.vis)return;const col=flowerOn?clsColor(r.cls):CROWN_COLOR,on=(i===sel&&mode==='select'),p=r.poly;
  polyPath(p);
  ctx.fillStyle=hexA(col,on?0.42:(predOn?0.10:0.22));ctx.fill();
  ctx.strokeStyle=on?'#ffffff':col;ctx.lineWidth=on?3:(predOn?1:2);ctx.stroke();
  if(on)for(const vi of handleIdx(p,HANDLE_PX)){const k=vi*2,hv=(k===hovPt);
   ctx.beginPath();ctx.arc(sx(p[k]),sy(p[k+1]),hv?7:5,0,7);ctx.fillStyle=hv?col:'#fff';ctx.fill();ctx.strokeStyle=hv?'#fff':col;ctx.lineWidth=hv?2:1.5;ctx.stroke();}});
 // Đỉnh giả: vòng tròn rỗng nét đứt + dấu cộng, để phân biệt ngay với đỉnh
 // thật (tròn đặc). Bấm vào là nó thành đỉnh thật.
 if(ghost&&mode==='select'&&sel>=0&&inst[sel]&&inst[sel].vis&&dragPt<0){
  const gx=sx(ghost.x),gy=sy(ghost.y);
  ctx.beginPath();ctx.arc(gx,gy,6,0,7);
  ctx.fillStyle='rgba(255,255,255,.30)';ctx.fill();
  ctx.setLineDash([3,3]);ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();ctx.moveTo(gx-3.5,gy);ctx.lineTo(gx+3.5,gy);
  ctx.moveTo(gx,gy-3.5);ctx.lineTo(gx,gy+3.5);
  ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();}
 if(draft.length){ctx.strokeStyle='#f97316';ctx.setLineDash([6,4]);ctx.beginPath();
  for(let k=0;k<draft.length;k+=2){k===0?ctx.moveTo(sx(draft[k]),sy(draft[k+1])):ctx.lineTo(sx(draft[k]),sy(draft[k+1]));}
  if(mouse)ctx.lineTo(sx(cX(mouse[0])),sy(cY(mouse[1])));ctx.stroke();ctx.setLineDash([]);
  for(let k=0;k<draft.length;k+=2){ctx.beginPath();ctx.arc(sx(draft[k]),sy(draft[k+1]),4,0,7);ctx.fillStyle='#f97316';ctx.fill();}}
 if(mode==='sam-box'&&boxStart&&mouse){const bx=cX(boxStart[0]),by=cY(boxStart[1]),mx=cX(mouse[0]),my=cY(mouse[1]);ctx.strokeStyle='#f97316';ctx.setLineDash([5,4]);ctx.strokeRect(sx(bx),sy(by),(mx-bx)*scale,(my-by)*scale);ctx.setLineDash([]);}}});
tabRegister({id:'crown',inspTitle:'Chi tiết',
 help:[['V','Xem (nhãn tán không sửa được)'],['C · B · D · S','Click · Box · Vẽ tay · Nắn (khi Sửa)'],['Shift + kéo đỉnh','Chỉ nắn đúng một đỉnh'],
  ['Chuột phải lên đỉnh','Xoá đỉnh đó'],['Rê lên cạnh','Hiện điểm ⊕, bấm vào là thêm đỉnh và kéo luôn'],['Nháy đúp','Đóng vùng khi Vẽ tay'],
  ['Del','Xoá tán đang chọn'],['Esc','Huỷ nét đang vẽ'],['Ctrl Z','Hoàn tác']],
 // Hai phím Ctrl+Z và Delete SỬA nhãn nên cũng phải khoá khi đang Xem hoặc xem
 // dự đoán — chúng đi thẳng vào `inst` chứ không qua handler chuột.
 key(e){if((e.ctrlKey||e.metaKey)&&(e.key==='z'||e.key==='Z')){e.preventDefault();if(!predLocked())undo();return true;}
  if(e.ctrlKey||e.metaKey||e.altKey)return false;
  if(e.key==='Escape'){draft=[];draw();return true;}
  if(e.key==='Delete'){if(!predLocked()&&sel>=0&&inst[sel]&&inst[sel].vis){pushHist();inst.splice(sel,1);sel=-1;refresh();}return true;}
  const k=e.key.toLowerCase();
  if(k==='v'){setMode('view');return true;}
  const m={c:'sam-click',b:'sam-box',d:'draw',s:'select'}[k];
  if(m&&mode!=='view'&&!predOn){setMode(m);return true;}
  return false;}});
