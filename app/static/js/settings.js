// settings.js — nguồn dữ liệu, model SAM, nơi lưu nhãn.
async function loadFolders(){try{const d=await(await fetch('/api/folders',{cache:'no-store'})).json();
 curFolder=d.current||'';const s=$('folder');s.innerHTML='';
 const o0=document.createElement('option');o0.value='';o0.textContent='📁 Tất cả các field';s.appendChild(o0);
 (d.folders||[]).forEach(f=>{const o=document.createElement('option');o.value=f;o.textContent='📁 '+f;s.appendChild(o);});
 s.value=curFolder;}catch(e){}}
async function loadModels(){const s=$('modelSel');if(!s)return;
 try{const d=await(await fetch('/api/models',{cache:'no-store'})).json();s.innerHTML='';
  (d.models||[]).forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=(m.variant==='sam3'?'✦3 ':'✦ ')+m.name;if(m.id===d.current)o.selected=true;s.appendChild(o);});
  if(!(d.models||[]).length){const o=document.createElement('option');o.textContent='(không có model trong weights/)';s.appendChild(o);}}
 catch(e){}}
// Nơi autosave nhãn là cố định theo configs/config.yaml — chỉ đọc để hiện lên
// tooltip chỉ báo lưu. Nơi xuất thì hỏi lại mỗi lần, đây chỉ là thư mục mở đầu.
let saveDirStart='',expDirStart='';
async function loadPaths(){try{const d=await(await fetch('/api/paths',{cache:'no-store'})).json();
 saveDirStart=d.out_dir||'';expDirStart=d.export_dir||'';
 svst('','','Nhãn lưu tại: '+(saveDirStart||'(mặc định)'));
 $('setPaths').textContent='Nhãn tán: '+(saveDirStart||'—')+' · xuất mặc định: '+(expDirStart||'—');
 $('dataPaths').textContent='Nhãn tán lưu tại '+(saveDirStart||'—')+' · bộ xuất mặc định ở '+(expDirStart||'—')+'.';}catch(e){}}
// Tên model SAM đang chọn, hiện nhỏ cạnh công cụ sửa tán.
function syncSamName(){const s=$('modelSel'),o=s&&s.options[s.selectedIndex];$('samName').textContent=o?o.textContent.replace(/^✦3? /,''):'';}
// Người gán: chọn ở thanh trên, thêm/bớt trong ⚙. Gửi kèm mọi thao tác gán hoa.
let annotators=lsGet('cofseg_annotators',['A','B']),annot=lsGet('cofseg_annot','A');
function renderAnnot(){if(!annotators.includes(annot))annot=annotators[0];
 [$('annotBox'),$('annotList')].forEach(box=>{box.querySelectorAll('button').forEach(b=>b.remove());
  annotators.forEach(n=>{const b=document.createElement('button');b.textContent=n;b.className=(n===annot?'on':'');
   b.title='Người gán '+n;b.setAttribute('aria-pressed',n===annot);
   b.onclick=()=>{annot=n;lsSet('cofseg_annot',n);renderAnnot();};box.appendChild(b);});});}
function addAnnot(){const n=($('annotNew').value||'').trim().toUpperCase().slice(0,4);
 if(!n||annotators.includes(n))return;annotators.push(n);lsSet('cofseg_annotators',annotators);$('annotNew').value='';renderAnnot();}
