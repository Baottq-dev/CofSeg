// core.js — phần dùng chung của annotator: canvas, khung nhìn, nạp ảnh,
// điều hướng, thanh trạng thái và sổ đăng ký các phần giao diện.
// Các file JS là script thường, nạp theo thứ tự trong index.html và dùng chung
// phạm vi toàn cục. File chỉ ĐỊNH NGHĨA; gắn sự kiện và khởi động nằm ở main.js.
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),wrap=document.getElementById('wrap');
const $=id=>document.getElementById(id);
function enc(t){return t.split('/').map(encodeURIComponent).join('/');}
function natcmp(a,b){return String(a).localeCompare(String(b),undefined,{numeric:true});}
let imgs=[],labeledSet=new Set(),current=null,img=new Image(),curFolder='',pathFilter='';
let baseFit=1,zoom=1,scale=1,ox=0,oy=0;
let selToken=0,ready=false;
// --- sổ đăng ký -------------------------------------------------------------
// Mỗi phần giao diện (HSV/Otsu, sửa tán, dự đoán…) đăng ký một đối tượng có
// các móc tuỳ chọn; core gọi chúng theo thứ tự đăng ký. Thứ tự đó cũng là thứ
// tự VẼ: lớp pixel hoa, rồi tán, rồi dự đoán đè lên trên cùng.
//   beforeImage()        trước khi rời ảnh đang mở (lưu nốt phần đang sửa)
//   resetImage(t)        dọn trạng thái của ảnh cũ, đồng bộ
//   loadImage(t, tok)    nạp dữ liệu của ảnh mới (async); tok đổi là bỏ
//   imageShown()         ảnh đã tải xong và khung nhìn đã đặt lại
//   onImage(t, tok)      dữ liệu đã sẵn sàng (ready = true)
//   draw(ctx)            vẽ lớp của mình lên canvas
const WS=[];
function wsRegister(w){WS.push(w);return w;}
function wsEach(name,...args){for(const w of WS)if(w[name])w[name](...args);}
// --- lưu tạm theo trình duyệt (cột, thẻ, bộ lọc…): hỏng thì dùng mặc định ----
function lsGet(k,d){try{const v=localStorage.getItem(k);return v==null?d:JSON.parse(v);}catch(e){return d;}}
function lsSet(k,v){try{localStorage.setItem(k,JSON.stringify(v));}catch(e){}}
// --- bốn thẻ: Tán · Hoa · So sánh · Dữ liệu ----------------------------------
// Mỗi thẻ đăng ký {id, inspTitle, help, enter(), leave(), key(e), mouse(loại,e)};
// key/mouse trả true nếu đã xử lý. Phần tử nào mang data-ws="…" chỉ hiện khi
// thẻ đang mở có tên trong danh sách đó. Đổi thẻ không đổi ảnh, không lưu gì.
const TABS={},TAB_ORDER=['crown','flower','review','data'];
const TAB_NAMES={crown:'Tán',flower:'Hoa',review:'So sánh',data:'Dữ liệu'};
let wsActive=null;
function tabRegister(t){TABS[t.id]=t;return t;}
function switchWs(id){if(!TABS[id])id='crown';if(id===wsActive)return;
 const old=wsActive&&TABS[wsActive];if(old&&old.leave)old.leave();
 wsActive=id;lsSet('cofseg_tab',id);
 document.querySelectorAll('#tabs button').forEach(b=>{const on=b.dataset.tab===id;b.classList.toggle('on',on);b.setAttribute('aria-selected',on);});
 document.querySelectorAll('[data-ws]').forEach(el=>{el.hidden=!el.dataset.ws.split(' ').includes(id);});
 $('inspTitle').textContent=TABS[id].inspTitle||'Chi tiết';
 layoutCols();
 if(TABS[id].enter)TABS[id].enter();
 renderStatus();}
function wsKey(e){const t=TABS[wsActive];return !!(t&&t.key&&t.key(e));}
function wsMouse(type,e){const t=TABS[wsActive];return !!(t&&t.mouse&&t.mouse(type,e));}
// Hai cột bên thu gọn được; thẻ Dữ liệu không có cột bên. Thu/mở xong thì
// canvas đổi cỡ nhưng giữ nguyên mức phóng và vị trí đang xem.
let colState=lsGet('cofseg_cols',{nav:true,insp:true});
function layoutCols(){const sides=wsActive!=='data';
 $('navcol').hidden=!(sides&&colState.nav);$('navrail').hidden=!(sides&&!colState.nav);
 $('inspcol').hidden=!(sides&&colState.insp);$('insprail').hidden=!(sides&&!colState.insp);
 document.querySelectorAll('.resizer').forEach(r=>{r.hidden=$(r.dataset.t).hidden;});
 if(sides&&img.width){sizeCanvas();if(scale>0&&isFinite(scale))draw();else resetView();}}
function toggleCol(which){colState[which]=!colState[which];lsSet('cofseg_cols',colState);layoutCols();}
// Thanh trạng thái: ảnh đang mở; số tán chỉ có nghĩa ở thẻ Tán.
function renderStatus(){const t=current?current.split('/'):[];
 $('stWhere').textContent=(wsActive==='data'||!current)?'':((t.length>1?t[0]+' › ':'')+t[t.length-1]);
 $('count').hidden=(wsActive!=='crown');}
// Bảng phím: phím của thẻ đang mở, rồi phím dùng chung.
const HELP_GLOBAL=[['Alt 1…4','Chuyển thẻ Tán · Hoa · So sánh · Dữ liệu'],['PgUp · PgDn','Ảnh trước · sau trong danh sách đang lọc'],
 ['Lăn chuột · kéo','Phóng to, thu nhỏ · di chuyển ảnh'],['+ · − · 0','Phóng to · thu nhỏ · vừa khung'],['?','Bảng phím này'],['Esc','Đóng hộp thoại đang mở']];
function openHelp(){const t=TABS[wsActive]||{};$('helpTitle').textContent='⌨ Phím tắt · thẻ '+TAB_NAMES[wsActive];
 const box=$('helpList');box.innerHTML='';
 [...(t.help||[]),null,...HELP_GLOBAL].forEach(r=>{const d=document.createElement('div');
  if(!r){d.style.height='6px';box.appendChild(d);return;}
  const k=document.createElement('b');k.className='kbd';k.textContent=r[0];
  const s=document.createElement('span');s.textContent=r[1];d.appendChild(k);d.appendChild(s);box.appendChild(d);});
 $('helpModal').classList.add('open');}
function openSettings(){$('setModal').classList.add('open');}
// Esc đóng hộp thoại trên cùng. Cửa sổ chọn thư mục đang chờ trả lời thì để
// nó tự đóng bằng nút Huỷ, không đóng ngầm ở đây.
function closeModals(){for(const id of ['cfModal','helpModal','setModal']){if($(id).classList.contains('open')){$(id).classList.remove('open');return true;}}return false;}
// --- điều hướng: một cột, lọc theo ruộng / chữ / trạng thái, nhóm theo đường bay ---
let navFilter=lsGet('cofseg_navfilter','all'),navThumbs=lsGet('cofseg_navthumbs',false);
let collapsed=new Set(lsGet('cofseg_navcollapsed',[]));   // nhóm đường bay đang gập
function isDone(p){return labeledSet.has(p);}
function inScope(p){return pathFilter===''||p===pathFilter||p.indexOf(pathFilter+'/')===0;}
function fieldOf(p){const i=p.indexOf('/');return i<0?'':p.slice(0,i);}
function groupOf(p){const i=p.lastIndexOf('/');return i<0?'':p.slice(0,i);}
// Tên ảnh DJI dài và giống nhau tới gần hết, bị cắt "…" thì không còn phân biệt
// được: hiện số khung và giờ chụp, tên đầy đủ để ở tooltip.
function frameLabel(p){const f=p.split('/').pop(),m=f.match(/^DJI_\d{8}(\d\d)(\d\d)(\d\d)_(\d{4})_/);
 return m?(m[4]+' · '+m[1]+':'+m[2]+':'+m[3]):f;}
function navList(){const q=($('search').value||'').toLowerCase();
 return imgs.filter(p=>inScope(p)&&p.toLowerCase().indexOf(q)>=0&&(navFilter==='all'||(navFilter==='done')===isDone(p)));}
// Số đếm bám theo ruộng đang chọn: đang gán dở một ruộng thì cần biết còn bao
// nhiêu ảnh của RUỘNG ĐÓ, chứ tổng toàn bộ không giúp gì.
function renderStats(){
 const list=pathFilter?imgs.filter(inScope):imgs;
 const total=list.length,lab=list.filter(isDone).length;
 $('stTotal').textContent=total;$('stLab').textContent=lab;
 $('stUnlab').textContent=Math.max(0,total-lab);$('stCls').textContent=classes.length;
 $('stTotalLbl').textContent='Ảnh'+(pathFilter?(' · '+pathFilter):' · tất cả');
 $('nfAll').textContent=total;$('nfTodo').textContent=Math.max(0,total-lab);$('nfDone').textContent=lab;}
async function loadLabeled(){try{const d=await(await fetch('/api/labeled?t='+Date.now(),{cache:'no-store'})).json();labeledSet=new Set(d.labeled||[]);emptySet=new Set(d.empty||[]);}catch(e){labeledSet=new Set();emptySet=new Set();}renderStats();}
function firstImgIn(pre){return imgs.find(p=>pre===''||p===pre||p.indexOf(pre+'/')===0);}
function pickDir(full){pathFilter=full;renderPTree();renderThumbs();renderStats();
 const f=firstImgIn(full);if(f&&f!==current)selectImg(f);}
// Ô chọn ruộng thay cho cây thư mục. Giữ tên renderPTree cho các chỗ gọi cũ.
function renderPTree(){const s=$('navField');const fs=[...new Set(imgs.map(fieldOf).filter(Boolean))].sort(natcmp);
 s.innerHTML='';const o0=document.createElement('option');o0.value='';o0.textContent='Tất cả · '+imgs.length+' ảnh';s.appendChild(o0);
 fs.forEach(f=>{const n=imgs.filter(p=>fieldOf(p)===f).length,o=document.createElement('option');
  o.value=f;o.textContent=f+' · '+n+' ảnh';s.appendChild(o);});
 s.value=fs.includes(pathFilter)?pathFilter:'';}
function setNavFilter(f){navFilter=f;lsSet('cofseg_navfilter',f);
 document.querySelectorAll('#navFilter button').forEach(b=>b.classList.toggle('on',b.dataset.f===f));renderThumbs();}
function setNavThumbs(on){navThumbs=on;lsSet('cofseg_navthumbs',on);$('navThumbs').checked=on;renderThumbs();}
// Danh sách ảnh nhóm theo đường bay; đầu nhóm gập/mở được và mang số đã gán.
// Vẽ lại thì giữ vị trí cuộn, chỉ cuộn khi ảnh đang mở đổi.
function renderThumbs(){const box=$('thumbs'),st=box.scrollTop;box.innerHTML='';box.classList.toggle('noimg',!navThumbs);
 const list=navList();
 $('nimg').textContent=(list.length===imgs.length)?imgs.length:(list.length+' / '+imgs.length);
 const tally={};imgs.forEach(p=>{const g=groupOf(p),t=tally[g]||(tally[g]={n:0,done:0});t.n++;if(isDone(p))t.done++;});
 let g=null,open=false;
 list.forEach(p=>{const k=groupOf(p);
  if(k!==g){g=k;open=!collapsed.has(k);
   const h=document.createElement('div');h.className='ghead';
   const nm=document.createElement('b');nm.textContent=(open?'▾ ':'▸ ')+(k||'(gốc)');nm.style.fontWeight='600';
   const c=document.createElement('span');c.textContent=tally[k].done+'/'+tally[k].n;
   h.appendChild(nm);h.appendChild(c);h.title=(k||'(gốc)')+' — đã gán '+tally[k].done+'/'+tally[k].n+' ảnh';
   h.onclick=()=>{if(collapsed.has(k))collapsed.delete(k);else collapsed.add(k);lsSet('cofseg_navcollapsed',[...collapsed]);renderThumbs();};
   box.appendChild(h);}
  if(!open)return;
  const row=document.createElement('div');row.className='thumb'+(p===current?' active':'')+(isDone(p)?' done':'')+(emptySet.has(p)?' empty':'');row.dataset.n=p;
  if(navThumbs){const im=document.createElement('img');im.loading='lazy';im.src='/api/thumb/'+enc(p);row.appendChild(im);}
  const nm=document.createElement('div');nm.className='tname';nm.textContent=frameLabel(p);nm.title=p;
  row.appendChild(nm);row.onclick=()=>selectImg(p);box.appendChild(row);});
 box.scrollTop=st;}
let lastActive=null;
function updateThumbActive(){document.querySelectorAll('#thumbs .thumb').forEach(t=>{t.classList.toggle('active',t.dataset.n===current);t.classList.toggle('done',isDone(t.dataset.n));t.classList.toggle('empty',emptySet.has(t.dataset.n));});
 if(current!==lastActive){lastActive=current;const a=document.querySelector('#thumbs .thumb.active');if(a&&a.scrollIntoView)a.scrollIntoView({block:'nearest'});}}
// PgUp/PgDn: ảnh trước/sau theo đúng danh sách đang lọc trên màn hình.
function stepImg(d){const list=navList();if(!list.length)return;
 let i=list.indexOf(current);i=(i<0)?0:Math.min(list.length-1,Math.max(0,i+d));
 if(list[i]!==current)selectImg(list[i]);}
async function loadImgs(){imgs=await(await fetch('/api/images',{cache:'no-store'})).json();
 await loadLabeled();renderPTree();renderThumbs();renderStats();
 // Mở lại ảnh lần trước nếu nó còn trong danh sách.
 const last=lsGet('cofseg_img',null);
 if(imgs.length)selectImg(imgs.includes(last)?last:imgs[0]);else{inst=[];current=null;draw();renderRegions();}}
// Đổi ảnh. `tok` chốt lần đổi: hai lần đổi chồng nhau thì lần về sau không
// được ghi đè bằng kết quả của ảnh trước (mọi phần nạp dữ liệu đều hỏi lại).
async function selectImg(t){wsEach('beforeImage');
 const tok=++selToken;
 current=t;ready=false;lsSet('cofseg_img',t);renderStatus();wsEach('resetImage',t);updateThumbActive();
 img=new Image();img.onload=()=>{if(tok===selToken){sizeCanvas();resetView();wsEach('imageShown');}};
 img.src='/api/image/'+enc(t)+'?t='+Date.now();
 for(const w of WS){if(!w.loadImage)continue;await w.loadImage(t,tok);if(tok!==selToken)return;}
 ready=true;wsEach('onImage',t,tok);draw();renderStatus();}
// Chỉ báo lưu. Bản cũ gọi fetch rồi bỏ qua kết quả, nên server có báo lỗi cũng
// không ai thấy — kể cả khi nó không đọc được ảnh và không tính được mật độ hoa.
function svst(cls,txt,title){const el=$('svst');if(!el)return;
 el.className=cls||'';el.textContent=txt||'';el.title=title||txt||'';}
function hhmm(){const d=new Date(),p=n=>('0'+n).slice(-2);
 return p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());}
function sizeCanvas(){cv.width=wrap.clientWidth;cv.height=wrap.clientHeight;}
function resetView(){if(!img.width)return;baseFit=Math.min(cv.width/img.width,cv.height/img.height,1);zoom=1;scale=baseFit;ox=(cv.width-img.width*scale)/2;oy=(cv.height-img.height*scale)/2;setZlbl();draw();}
function zoomAt(cx,cy,f){if(!img.width)return;const ix=(cx-ox)/scale,iy=(cy-oy)/scale;zoom=Math.min(12,Math.max(0.1,zoom*f));scale=baseFit*zoom;ox=cx-ix*scale;oy=cy-iy*scale;setZlbl();draw();}
function setZlbl(){$('zlbl').textContent=Math.round(zoom*100)+'%';}
function toImg(e){const r=cv.getBoundingClientRect();const kx=r.width?cv.width/r.width:1,ky=r.height?cv.height/r.height:1;return[((e.clientX-r.left)*kx-ox)/scale,((e.clientY-r.top)*ky-oy)/scale];}
// Kẹp toạ độ về trong khung ảnh: kéo/vẽ tràn ra ngoài thì điểm dán vào mép
// chứ không nằm ngoài ảnh. Biên [0,w]/[0,h] khớp đúng _clip_poly bên server.
function cX(x){return img.width?Math.min(Math.max(x,0),img.width):x;}
function cY(y){return img.height?Math.min(Math.max(y,0),img.height):y;}
function sx(x){return x*scale+ox;}
function sy(y){return y*scale+oy;}
function hexA(h,a){const n=parseInt(h.slice(1),16);return'rgba('+(n>>16&255)+','+(n>>8&255)+','+(n&255)+','+a+')';}
function centroid(p){let x=0,y=0,n=p.length/2;for(let i=0;i<p.length;i+=2){x+=p[i];y+=p[i+1];}return[x/n,y/n];}
function polyPath(p){ctx.beginPath();
 for(let k=0;k<p.length;k+=2){k===0?ctx.moveTo(sx(p[k]),sy(p[k+1])):ctx.lineTo(sx(p[k]),sy(p[k+1]));}
 ctx.closePath();}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);
 if(img.complete&&img.width)ctx.drawImage(img,ox,oy,img.width*scale,img.height*scale);
 for(const w of WS)if(w.draw)w.draw(ctx);}
function pip(p,x,y){let c=false;for(let i=0,j=p.length-2;i<p.length;j=i,i+=2){const xi=p[i],yi=p[i+1],xj=p[j],yj=p[j+1];if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))c=!c;}return c;}
// Phím tắt chỉ dành cho canvas. Đang gõ trong ô tìm kiếm / ô số Otsu / dropdown
// lớp thì trả phím về cho ô đó: trước đây gõ "0" là reset zoom, "-" là zoom
// out, Delete là xoá vùng đang chọn (kể cả vùng đang ẩn, không thấy nó mất).
function typing(e){const t=e.target,tag=t&&t.tagName;
 return tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT'||(t&&t.isContentEditable);}
// Hộp xác nhận trong trang thay cho confirm(): confirm() chặn cả luồng sự kiện
// của trình duyệt, và không đặt được nút "Huỷ" là mặc định.
function askConfirm(o,onOk){const m=$('cfModal'),b=$('cfOk');
 $('cfTitle').textContent=o.title||'';$('cfText').textContent=o.text||'';
 b.textContent=o.ok||'Đồng ý';b.className='cfok'+(o.danger?' danger':'');
 const done=v=>{m.classList.remove('open');b.onclick=$('cfCancel').onclick=m.onclick=null;if(v)onOk();};
 b.onclick=()=>done(true);$('cfCancel').onclick=()=>done(false);m.onclick=e=>{if(e.target===m)done(false);};
 m.classList.add('open');$('cfCancel').focus();}
// Phóng tới một polygon (toạ độ ảnh), chừa lề quanh nó.
function zoomToPoly(p,pad){if(!img.width||!p||p.length<6)return;
 let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
 for(let k=0;k<p.length;k+=2){x0=Math.min(x0,p[k]);x1=Math.max(x1,p[k]);y0=Math.min(y0,p[k+1]);y1=Math.max(y1,p[k+1]);}
 const bw=Math.max(40,x1-x0),bh=Math.max(40,y1-y0),f=pad||1.6;
 const s=Math.min(cv.width/(bw*f),cv.height/(bh*f));
 zoom=Math.min(12,Math.max(0.1,s/baseFit));scale=baseFit*zoom;
 ox=cv.width/2-(x0+x1)/2*scale;oy=cv.height/2-(y0+y1)/2*scale;setZlbl();draw();}
