document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('themeToggle');
  btn && btn.addEventListener('click', () => {
    const body = document.body;
    const isDark = body.classList.toggle('dark');
    if(isDark) document.documentElement.classList.add('dark'); else document.documentElement.classList.remove('dark');
    const theme = isDark ? 'dark' : 'light';
    document.cookie = `theme=${theme}; path=/; max-age=${60*60*24*365}`;
    fetch('/theme', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({theme})
    }).catch(()=>{});
  });

  // Career search/filter
  const search = document.getElementById('careerSearch');
  if(search){
    search.addEventListener('input', function(e){
      const q = e.target.value.toLowerCase();
      document.querySelectorAll('.career-card-wrapper').forEach(function(wrapper){
        const card = wrapper.querySelector('.career-card');
        const name = (card.dataset.career || '').toLowerCase();
        wrapper.style.display = name.includes(q) ? '' : 'none';
      });
    });
  }

  // Modal details for career cards
  const careerModalEl = document.getElementById('careerModal');
  let careerModal = careerModalEl ? new bootstrap.Modal(careerModalEl) : null;
  document.querySelectorAll('.view-details').forEach(function(btn){
    btn.addEventListener('click', function(){
      const card = btn.closest('.career-card');
      if(!card) return;
      const title = card.dataset.career || '';
      const roadmap = (card.dataset.roadmap || '').split('||').filter(Boolean);
      const skills = card.dataset.skills || '';
      const salary = card.dataset.salary || '';

      document.getElementById('careerModalLabel').textContent = title;
      document.getElementById('careerModalExplanation').textContent = card.querySelector('.card-text')?.textContent || '';
      const ul = document.getElementById('careerModalRoadmap');
      ul.innerHTML = '';
      roadmap.forEach(function(item){ const li = document.createElement('li'); li.textContent = item; ul.appendChild(li); });
      document.getElementById('careerModalSkills').textContent = skills;
      document.getElementById('careerModalSalary').textContent = salary;
      careerModal && careerModal.show();
    });
  });
});
