# Отчёт агента 2 — 14 октября 2026

## Опора пола и купон

Выполнены два физически рассчитанных capture перенесённого купона на внутреннюю
часть конечного пола: по 1 680 шагов (`dt=1 ms`, 4 подшага) и 177 настоящих
состояний от 0 до 1,68 s. PNG/GIF/MP4 создавались из сохранённых состояний.

- Первый, CUDA baseline: [CSV](validation/agent2-support-cuda.csv),
  [GIF](validation/agent2-support-cuda.gif), [MP4](validation/agent2-support-cuda.mp4).
  На RTX 5060 Laptop через WSL было 13 082 вызова CUDA PCG; 12 724 завершились
  выше tolerance `1e-5`, максимум residual `9.716e-3`, финальный `2.966e-4`.
  Старый бинарь всё равно применял эти поправки, поэтому результат является
  диагностическим baseline, а не физической валидацией CUDA.
- Второй, CPU fallback: [CSV](validation/agent2-support-cuda-cpu-fallback.csv),
  [GIF](validation/agent2-support-cuda-cpu-fallback.gif),
  [MP4](validation/agent2-support-cuda-cpu-fallback.mp4). В промежуточном бинаре
  CUDA завершилась ошибкой `cuMemcpyDtoH(dot partials)`, solver продолжил расчёт
  на CPU. CPU residual оставался ниже `1e-5`.

В CUDA baseline минимальная нижняя грань купона была `+3.62e-7 m`, в последнем
состоянии `+3.86e-7 m`. В CPU fallback capture минимальный зазор составил
`−9.46e-7 m` (менее микрометра, в пределах допуска проекции), финальный —
`+1.27e-7 m`. В обоих capture основание дерева оставалось на `z=0`, дрейф
направляющей равен нулю. Ствол не получил разрывов связей; купон частично
раскололся: 32 уникальные связи в CUDA baseline, 36 в CPU fallback. Это
подтверждает поддержку пола с указанной численной погрешностью, но не
сохранность купона.

Контактное тепловыделение `EnergyLedger::contact_friction_heat` достигло
`0.001211 J` в CUDA baseline и `0.004507 J` в CPU fallback. Capture не содержит
полного баланса кинетической, внутренней, упругой и граничной энергии; полный
энергетический баланс этих прогонов не заявляется.

Оба MP4 проверены через ffprobe: 177 кадров, 10 fps, длительность 17,7 s,
H.264 1600×1000. Первый GIF имеет 1280×800, второй GIF после обновления
renderer — 1600×1000; оба содержат 177 кадров. В момент проверки viewer отвечал
HTTP 200 на `/gif` и `/video`. Результаты и пределы подробней описаны в
[диагностике пола](validation/floor-support-diagnosis.md).

## CUDA kernels

`validation/cuda_sparse_probe.py` расширен: он проверяет sparse matvec, block
Jacobi, направление PCG, dot reduction и scalar ratio, включая размеры 4 203 и
10 000 REV. На RTX 5060 синтетические проверки прошли: максимальная ошибка
matvec `6.104e-5`, Jacobi `5.960e-8`, ошибка направления `0`, относительная
ошибка dot `0`, scalar ratio `4.5 / 1.5 = 3`. Агент 1 также сообщил результат
синтетической двухузловой device-resident PCG: 2 итерации, residual
`6.104e-8`, ошибка поправки `7.827e-8`. Эти kernel/operator проверки не являются
физическим GPU benchmark.

Обнаруженные в ходе координации измерения относятся к разным бинарям. Read-only
сравнение реализации показало, что CPU применяет IC(0) (`build_ic0` и
`apply_ic0`), а CUDA — блочный Якоби; одинаковые 80 итераций не означают
одинаковую скорость сходимости. Также найден множитель 256 в прежнем `dot_reduce`:
все 256 потоков суммировали один и тот же список partials. Агент 1 исправил
смещение на `tid` и добавил в probe проверку 17 ненулевых partials. Проверка на
RTX 5060 дала сумму 153 для 17 partials; synthetic PCG после всех 80 фиксированных
итераций имеет residual `4.131e-17`. Запись PID 6263, собранная до этого фикса,
остановилась на шаге 955/1680 и считается неполной. В профилированной записи из
исправленного бинаря первые отдельные CUDA solves на 4 203 REV сошлись до
`residual≈1.4e-6`, но лог также показывает отказ CUDA PCG до разрушения связей
(`8.656e-5` при допуске `1e-5`) и CPU fallback. По всем шагам CPU выполнял
значительную часть PCG; capture является смешанным CUDA/CPU, а не GPU-only
валидацией.
Новый capture завершён: 1 680 шагов, 177 состояний, `dt=1 ms`, 4 подшага,
примерно 8 min wall time (около 3,4 шага/с). PNG/GIF/MP4 созданы из всех
сохранённых состояний; GIF — 1600×1000, 177 кадров; MP4 — H.264 1600×1000,
177 кадров, 10 fps, 17,7 s, проверен полным декодированием. Минимум нижней грани
купона `−5.57e-7 m`, финальный зазор `+5.16e-8 m`; дрейф направляющей 0. Купон
разрушил 36 уникальных связей; в финале 10 компонент; контактное тепло
`0.002738 J`.

После исправления dot reduction первые CUDA solves на полной целой системе
достигли residual около `1.4e-6` при tolerance `1e-5`. Связи купона рвутся на
шаге 296; в более поздних выводах наблюдается CPU PCG, что указывает на fallback
при изменённой топологии. Точную ошибку в stdout не сохранили, поэтому устойчивую
GPU-сходимость после разрушения пока не подтверждаем. Цели 30 FPS CPU и 60 FPS
GPU не достигнуты.

## Проверки

- `python3 validation/cuda_sparse_probe.py 4203 1 1` и
  `python3 validation/cuda_sparse_probe.py` прошли для доступных kernels.
- `git diff --check` прошёл.
- Цели 30 FPS CPU и 60 FPS GPU не достигнуты и не подтверждены.

## Диагностика fallback — дополнение

Для диагностического capture `agent2-support-cuda-postfix` был временно собран
профилирующий вариант решателя: он печатает каждую причину CUDA fallback.
Временную правку исходника затем откатил, поскольку задание агента 2 запрещает
изменять `src/solver.rs`/CUDA. Точный лог сохранён рядом как
`validation/agent2-support-cuda-profiled.log`; результаты сообщены агенту 1 в
`chat.md`. Причина уже видна до разрушения связей: CUDA PCG исчерпывает 80
итераций с относительным residual `8.656e-5` и затем `1.210e-4` при допуске
`1e-5`. На изменённой топологии лог ещё собирается. Это указывает на
недостаточный сходимый бюджет CUDA PCG с текущим предобуславливателем, но
проверку большего бюджета или иного предобуславливателя не выполнял.

Профилирующая сборка проверена: `cargo test --lib` — 34 теста прошли;
`cargo fmt --check` и `git diff --check` прошли. Эти проверки предшествуют
текущим незакоммиченным изменениям агента 1 в `src/scene.rs`, `src/solver.rs`,
`src/cuda_solver.rs` и `MECHANICS_EXPERIMENTS.md`.

### Завершённый профилированный повтор

Команда: `VOXEL_PROFILE_STEP=1 VOXEL_SOLVER_BACKEND=cuda RAYON_NUM_THREADS=1
target/release/support_capture validation/agent2-support-cuda-postfix.csv 1680
177`. Сцена: 4 347 REV, 4 203 активных, `fixed_dt=1 ms`, 4 подшага,
80 PCG-итераций, RTX 5060 Laptop через WSL; CPU — Intel Core i5-14450HX.
Предыдущие CSV/GIF/MP4 сохранены в файлах с суффиксом `-prior`.

CSV подтверждён: 1 680 шагов, 177 реальных состояний, 769 419 REV-записей.
Первый CUDA fallback наступил примерно около шага 90, когда связей ещё не было
сломано. Всего профиль сохранил 360 успешных CUDA вызовов и CUDA fallback на 1 591
fixed step; один шаг может содержать несколько CPU PCG solves. Все fallback вызовы исчерпали 80 итераций: residual min/median/
max `2.531e-5` / `1.960e-4` / `2.608e-4` при допуске `1e-5`. Для успешных CUDA
вызовов median residual `1.429e-6`, максимум `7.393e-6`, median time `6.43 ms`.
После fallback CPU solve завершал fixed steps с median residual `8.639e-6`,
максимумом `9.999e-6`. На шаге 296 купон разрушил 36 связей; после этого CUDA
fallback residual держался около `1.9e-4`. Значит, причина CPU fallback —
недостаточная сходимость block-Jacobi PCG за фиксированные 80 итераций и на
целой системе под динамической нагрузкой, а не только перестройка после fracture.
Это локализует механизм отказа, но не проверяет, какой больший бюджет или
предобуславливатель надёжно восстановит CUDA-сходимость.

Медианные измерения `VOXEL_PROFILE_STEP` на один полный fixed step: predict/tool
31.20 ms; solve 193.91 ms (assembly 45.13, factor 3.58, PCG 127.15 ms);
collision 10.21 ms, fracture 5.25 ms, finish 16.34 ms. Медианная сумма этих
этапов — 256.91 ms/step. Полный GPU target 60 FPS не достигнут; это смешанный
CUDA/CPU прогон, не GPU-only benchmark.

Физический результат совпал с предыдущей полной записью: минимальная нижняя
грань купона `−5.566e-7 m`, финальная `+5.16e-8 m`; низ ствола `z=0`; drift
направляющей 0; купон потерял 36 связей и распался на 10 кластеров; contact
heat `0.002737991 J`. Renderer записал GIF и MP4 из всех 177 состояний; оба
видео имеют 1600×1000, 10 fps, 17.7 s. Viewer на 8765 отдаёт текущие CSV/GIF/MP4
с HTTP 200. Результаты и пределы сообщены агенту 1 в `chat.md`; изменения
текущего solver от агента 1 в этом отчёте не считаются проверенными.

## Задание агента 2: мировые границы

Задание из `TASKS_AGENT_WORLD_BOUNDS.md` выполнено. Сцена собрана публичным
`SceneBuilder`: шесть независимых железных REV, bounds `[-0.25, 0.25] m` по
каждой оси, без гравитации, `dt=0.01 s`, 2 подшага; 1 760 реальных шагов,
сохранение каждого десятого шага даёт 177 состояний за 17.6 s физического
времени. Ни capture, ни renderer не задают собственное движение после старта.

Команды воспроизведения:

```sh
cargo run --release --no-default-features --bin world_bounds_capture -- validation/world-bounds.csv
PYTHONPATH=.toolchain/python python3 validation/render_world_bounds.py validation/world-bounds.csv validation/world-bounds
```

Изменения задачи: `src/bin/world_bounds_capture.rs`,
`validation/render_world_bounds.py`, `validation/experiments.json`,
`MECHANICS_EXPERIMENTS.md`; записи `validation/world-bounds.csv`, `.gif`, `.mp4`.
В capture все 6 REV вышли через разные грани (по одному через X±, Y±, Z±); в
конце осталось 0, удалено 6. Ledger: масса `0.04721999354660511 kg`, внутренняя
энергия `9562.048706054688 J`, кинетическая энергия `7.879443340228773e-7 J`,
отложенная энергия связей `0 J`, импульс
`[6.296001270968176e-6, 7.870000214427533e-6, 7.870000214427533e-6] kg·m/s`.
Баланс массы и внутренней энергии — нулевая ошибка в записанной точности.
GIF/MP4 содержат 177 реальных кадров, 10 fps, 17.7 s, 1120×700; визуальных или
физических сбоев в этом опыте не зафиксировано. Viewer endpoints были проверены
с HTTP 200 после capture. Опыт подтверждает малую сцену и ledger, не
производительность крупных сцен.

#### Повторная проверка контакта после точного thermal accumulator

После добавления Агентом 1 `Voxel::add_internal_energy_j` контактные пути
restitution и friction используют его для распределения тепла в f64. Я
обновил проверки, чтобы считать энергию через `total_internal_energy_j()` и
сверять её с ledger с допуском `1e-12 J`, включая нормальный удар и скольжение.
На общем текущем worktree прошли полный набор
`VOXEL_SOLVER_BACKEND=cpu cargo test --no-default-features` (39 library,
5 physics smoke, 1 contact-energy, 1 CPU preconditioner, 3 fatigue topology,
6 supports; всего 55 тестов), `cargo check --release --no-default-features
--all-targets`, `cargo fmt --check`, `git diff --check`. Ambient physical capture
после исправления накопителя завершён Агентом 1 при 293.15 K; его capture
описан ниже и заменяет 1 K диагностическую запись.

#### Regression: fracture-generated fragment collision

Добавил `fatigue_generated_fragments_collide_through_solver_contact` в
`tests/fatigue_topology.rs`: циклическая нагрузка рвёт обе связи одного
трёх-REV образца и создаёт три cluster labels; через публичный `apply_impulse`
один созданный фрагмент запускается в другой, а `solver::step` обрабатывает
столкновение. Тест проверяет restitution heat, постоянство массы и импульса
между состоянием после импульса и после контакта, а также суммарную кинетическую
и внутреннюю энергию с абсолютной ошибкой ниже `1e-6 J`. Специализированный прогон и
полный набор CPU тестов прошли. Позднее Агент 1 записал длинный физический
capture той же последовательности; его параметры, drift и media validation
приведены ниже.
## Совместная CPU-запись агента 1 — промежуточная диагностика

Позднее процесс `support_capture validation/cpu-rayon2048-support.csv 1680 177` перестал существовать до завершения: CSV остановился на шаге 850 из 1 680 и содержит 90 из 177 состояний. Профильный лог заканчивается обычной строкой измерения, явной ошибки или причины остановки в нём нет. GIF/MP4 для этой записи отсутствуют; это незавершённый диагностический фрагмент, не физический capture и не итоговая скорость. Настройки из запуска и конфигурации: Intel Core i5-14450HX, CPU backend, 16 Rayon threads, 10 747 REV всего / около 4 203 механически активных, `dt=1 ms`, 4 подшага, cap 80 итераций; старая сборка использовала IC(0).

По 850 профилям медиана PCG составляет 867.29 ms/fixed step, 218 суммарных итераций и residual `8.425e-6`; после первых 100 профилей медиана составляет 898.85 ms, 219 итераций и residual `8.6595e-6`. Последний профиль имел 797.53 ms PCG и 194 итерации. Это промежуточный профиль одной траектории, не контролируемый A/B тест cutoff или IC(0)/BlockJacobi; цель CPU 30 FPS не достигнута. Позиция купона в старой сборке пересекала swept volume топора, поэтому запись не подтверждает гравитационную опору. Агент 1 обновил источник сцены на `y=+0.25 m`; требуется проверить физическое падение после отведения инструмента.

Viewer endpoint `8767` обслуживает текущую CSV; `8765` всё ещё показывает предыдущий CUDA capture. Агент 1 оставил текущую запись на отдельном viewer и не стал объявлять её видимой пользователю без ответа на вопрос о живом кадре.

### Короткая проверка исправленного положения купона

В следующем CSV `validation/support-contact-final.csv` купон действительно
перенесён в `y≈0.255 m`, вне диапазона инструмента. Это отдельная короткая
диагностика, не продолжение прерванной старой записи: CPU backend, 16 Rayon
потоков, `dt=1 ms`, 176 шагов и 177 реальных состояний, то есть только 0.176 s
физического времени. Начальный диапазон центров REV по Z — `0.075…0.095 m`.

По CSV нижняя грань впервые касается пола около шага 120; минимальный зазор за
все состояния — `+6.46e-8 m` на шаге 121. На шаге 176 нижний слой остался около
`z=0.005 m`, дрейф направляющей — 0. Купон при ударе частично разрушается:
18 разорванных связей, 4 кластера, контактное тепло `3.52165e-4 J`; верхний REV
в финале поднялся до центра `z=0.06755 m`. Capture подтверждает краткий контакт
с полом с малой погрешностью зазора, но не сохранность купона и не длительную
опору. Рендер завершён: `validation/support-contact-final.gif` и `.mp4`, 177
solver frames, 1600×1000, 10 fps, 17.7 s по `ffprobe`. Физическая длительность
сама симуляции — 0.176 s; это явно подписано в артефакте/отчёте и не равно
длительной физической проверке опоры. Кадровое воспроизведение замедлено примерно
в 101 раз относительно физического времени.

Старый рендерер показывал неверную фиксированную подпись «замедлено в 5 раз».
Агент 1 заменил её вычислением отношения media duration / physical duration.
Полная проверка мира и цели CPU/GPU производительности остаются открытыми.

### Контролируемый CPU cutoff A/B

Завершены два сопоставимых capture той же короткой support-сцены: default
`VOXEL_PARALLEL_MIN_VOXELS=2048` и serial override `32768`. Оба запускались на
Intel Core i5-14450HX, CPU backend, 16 Rayon threads, одном и том же
`target/release/support_capture`, `dt=1 ms`, 4 подшага, IC(0), 80 PCG iterations;
в начале сцена содержит 10 747 REV, около 4 203 механически активных. Оба CSV
имеют header `H,0.001,176,177`, а SHA-256 совпадает побайтно:
`adf8a17a0f25b14f865360979620d5ffed5f37ab998a590f391b68269b04a470`.

Для профилей после первых 50 шагов (после вступления в контакт) threshold 2 048
дал median PCG `1059.69 ms`, solve `1119.765 ms`, 304 итерации, residual median
`8.704e-6` и max `9.957e-6`. Serial threshold 32 768 дал PCG `126.225 ms` и
solve `182.04 ms`, с теми же 304 итерациями и теми же median/max residual.
PCG ускорился в `8.40×`, solve — в `6.15×`. Медианная сумма показанных в
профиле fixed-step стадий (predict/tool + solve + collision + fracture + finish)
снизилась с `1237.915` до `296.03 ms`; это сумма профилированных стадий, не
измерение полной wall-clock скорости capture с записью CSV и renderer.

Побайтно одинаковые CSV подтверждают идентичную траекторию для всех 177
состояний: 18 broken bonds, 4 кластера, min ground gap `+6.46e-8 m`, drift 0,
contact heat `3.52165e-4 J`. Следовательно, для этой сцены устранение Rayon
parallel overhead при 4 203 активных REV/16 потоках уменьшило время solve без
заметного изменения физики. Это короткий контактный прогон длиной 0.176 s;
полный длинный CPU benchmark и цель 30 FPS всё ещё не подтверждены.

### BlockJacobi full-scene диагностический capture

Агент 1 также записал `support-cpu-block-jacobi.csv` для той же 176-step сцены,
сменив только preconditioner на `block_jacobi`; CPU backend, cutoff 32 768,
16 Rayon threads, `dt=1 ms`, 4 подшага, cap 80. CSV имеет те же 177 состояний и
тот же начальный prelude, что serial IC(0), но SHA траекторий отличается.

После первых 50 шагов медиана PCG — `121.56 ms/fixed step`, solve `173.94 ms`,
агрегат 640 итераций, residual median `3.065e-4`, max `3.799e-4`. Подробный
профиль содержит 1 050 PCG calls: 692 (65.9%) завершаются выше tolerance
`1e-5`; многие доходят до cap 80. Медиана суммарных стадий профиля —
`284.675 ms`, но прямое сравнение latency до одинаковой точности некорректно,
потому что BlockJacobi не сошёлся на большинстве solve.

Диагностическая траектория всё же сохраняет 18 разорванных связей и 4 кластера;
max positional difference от сходившегося serial IC(0) — `1.1777e-4 m`, RMS
`1.778e-6 m`, минимум зазора купона `+6.30e-8 m`, drift 0, контактное тепло
`3.51984e-4 J`. Эти физические числа не подтверждают равную точность. Требуется
увеличить PCG cap до сходимости и повторить сравнение; при внесении этих данных
PNG/GIF/MP4 ещё рендерились, поэтому состояние media здесь не заявляется.

Для публичного выбора CPU preconditioner добавлен integration test
`tests/cpu_preconditioners.rs`: IC(0) и BlockJacobi прогоняют одну и ту же
нагруженную железную сцену с фиксированной гранью и частичным X-ограничением.
Оба решения должны укладываться в residual tolerance и совпадать по
координатам в пределах `2e-5 m`. Проверки: `VOXEL_SOLVER_BACKEND=cpu cargo test
--test cpu_preconditioners` — PASS (1/1); `VOXEL_SOLVER_BACKEND=cpu cargo test
--lib` — PASS (36/36); `rustfmt --edition 2021 --check
tests/cpu_preconditioners.rs` и `git diff --check` — PASS.

Рендер BlockJacobi capture завершён: `support-cpu-block-jacobi.gif/.mp4` имеют
177 кадров, 1600×1000, 10 fps и 17.7 s по ffprobe; MP4 полностью декодируется.
Viewer 8767 в момент проверки выбирал этот CSV и отдавал `/api/status`, `/gif`
и `/video` с HTTP 200. Как и для cutoff A/B, физическая длительность данной
короткой записи — 0.176 s; она не является полным длительным capture.

### BlockJacobi cap160: одинаковый критерий с IC(0)

Повтор `support-cpu-block-jacobi160` использует ту же сцену, CPU backend,
16 Rayon threads, cutoff 32 768, `dt=1 ms`, 4 подшага, но увеличенный PCG cap
160. Начальные header и prelude совпадают с serial IC(0), CSV имеет 177/177
состояний. Подробный лог содержит 1 049 solve calls: ни один не превысил
residual tolerance `1e-5`; median residual `9.351e-6`, максимум `9.999e-6`,
median 120 итераций, максимум 134.

После первых 50 шагов медианы BlockJacobi cap160 составили PCG `168.02 ms`,
solve `215.875 ms`, 963 агрегатные итерации, residual median/max
`9.0945e-6 / 9.997e-6`. Для serial IC(0) при тех же CPU/threads/scene и cutoff
медианы были PCG `126.225 ms`, solve `182.04 ms`, 304 итерации, residual
median/max `8.704e-6 / 9.957e-6`. По внутреннему PCG timer BlockJacobi примерно на
33% медленнее. Cap160 run включал вывод подробного текста после каждого PCG
solve; общий `solve` timer включает это дополнительное логирование, поэтому
`215.875` против `182.04 ms` нельзя считать чистым A/B solve-time.

Физические счётчики остались близкими: 18 разрывов и 4 кластера в обеих
записях. Разница max координаты — `1.0966e-4 m`, RMS `1.6421e-6 m`; минимум
зазора BJ `+6.37e-8 m`, drift 0, контактное тепло `3.5189952e-4 J` против
`3.5216538e-4 J` у IC(0). Физическая длительность — только 0.176 s. Рендер
завершён: `support-cpu-block-jacobi160.gif/.mp4`, каждый по 177 кадров,
1600×1000, 10 fps, 17.7 s; ffprobe подтвердил число кадров, MP4 полностью
декодируется. Viewer 8767 отдаёт `/gif` и `/video` с HTTP 200 и показывает CSV
этой записи. Длинный 17.6 s CPU capture и цель 30 FPS остаются открытыми.

### Повторная проверка engine-тестов — 14 октября

После проверки состояния worktree повторно запущены из текущего HEAD `db024bc`
изолированные engine regression tests: `VOXEL_SOLVER_BACKEND=cpu cargo test
--lib world_bounds_tests` — PASS (1/1), включая уплотнение всех REV-массивов,
индексов связей и отложенных событий после удаления на границе; `VOXEL_SOLVER_BACKEND=cpu
cargo test --lib thermal_transport_tests` — PASS (1/1), проверяет передачу тепла
между соединёнными горячим и холодным REV и сохранение энергии. Физический
захват не запускался, чтобы не менять активный viewer и не конкурировать с
работой Агента 1. Эти unit tests подтверждают локальные инварианты, но не
заменяют длинные физические записи из `MECHANICS_EXPERIMENTS.md`.

### Сквозная проверка усталостного разрушения — 14 октября

Добавлен тест `cyclic_subcritical_stress_breaks_bond_through_constitutive_update`
в `src/fracture.rs`. На железной связи повторная постоянная нагрузка 40% от
статической прочности не расходует усталостный ресурс; при реверсе той же
подкритической нагрузки constitutive `update` накапливает fatigue damage до 1,
разрывает bond, а энергия диссипации совпадает с доступной упругой энергией
шага разрушения в пределах `1e-6` относительной погрешности. Это проверяет соединение
счётчика циклов с публичным constitutive update, не является длинным физическим
захватом и не подтверждает сравнительную серию материалов. `cargo fmt --check`,
`VOXEL_SOLVER_BACKEND=cpu cargo test --lib fracture::tests` (11/11) и
`VOXEL_SOLVER_BACKEND=cpu cargo test --lib` (37/37), `git diff --check` прошли.

### Проверка свойств массового состава — 14 октября

Добавлен `material::tests::mass_fraction_mixture_preserves_specific_volume_and_heat_capacity`:
25/75 массовая смесь железа и меди должна иметь плотность
`1 / Σ(wᵢ/ρᵢ)`, удельную теплоёмкость `Σ(wᵢ cᵢ)`, а EOS должен вернуть 300 K из
энергии при этой температуре с полностью твёрдой фазой. Это regression для
свойств уже составного REV, не реализация физического слияния или образования
сплава. Точечный тест и полный library suite прошли; всего 38/38 lib tests.

### Проверка fatigue capture Агента 1 — 14 октября

Проверил завершённые артефакты: `fatigue-cycles.csv` содержит header
`H,0.01,1760,177,mechanical`, 177 записей `S` от 0 до 1760 и событие
`E,969,1,fatigue_fracture`. Лог сообщает max stress ratio `0.3804` и разрыв на
шаге 969 из 1 760; физическая длительность — 17.6 s. `ffprobe` насчитал 177
кадров в GIF и MP4; MP4 — 1600×1000, 10 fps, 17.7 s и декодируется. Viewer
`:8767` выбирает этот CSV и отдаёт `/api/status`, `/gif`, `/video` HTTP 200.
Viewer selector на `:8766` перечисляет `fatigue-cycles`; его статус показывает
тот же CSV и готовые медиа, запросы `/gif?experiment=fatigue-cycles`,
`/video?experiment=fatigue-cycles` и `/capture?...` также вернули HTTP 200.
Запись подтверждает одну усталостную сцену; сравнительный brittle/ductile опыт
остаётся незавершённым.

### Fatigue rupture и обновление связности solver — 14 октября

Добавлен integration test `tests/fatigue_topology.rs`:
`fatigue_failure_splits_clusters_and_released_rev_keeps_mass_and_motion` гоняет
двух-REV железную сцену через `solver::step_with_axis_targets` до усталостного
разрыва. Проверяет stress ratio ниже единицы, 2 разных cluster labels после
разрыва, сохранение суммарной массы, движение ранее фиксированного REV после
снятия нагрузки и сохранение этих инвариантов на следующем шаге. Команда
`VOXEL_SOLVER_BACKEND=cpu cargo test --test fatigue_topology` прошла (1/1,
0.09 s). Это проверяет топологическое последствие отказа, но не fragment impact,
массо-импульсный бюджет столкновений или произвольную сеть фрагментов.

### Solver pipeline: ledgered bond-release impulse — 14 октября

В `tests/fatigue_topology.rs` добавлен `bond_release_pipeline_conserves_momentum_and_ledgered_energy`.
Он кладёт queued bond release на два пространственно разобщённых железных REV,
проходит через `solver::step`, проверяет что `pending_release` списан, ledger
`kinetic_release` вырос на запрошенные 0.25 J, прирост кинетической энергии равен
0.25 J, суммарный импульс сохраняется с ошибкой ниже `1e-5 kg·m/s`, масса не
меняется. Вместе оба теста файла прошли 2/2. Это проверка API-пути применения
импульса, не тест столкновения двух физических кластеров.

### Учёт энергии нормального контакта — 14 октября

Исправлен пробел в контактной физике: нормальный импульс при restitution `e<1`
уменьшал кинетическую энергию, но потеря не попадала во внутреннюю энергию.
`collision::resolve_velocities` теперь вычисляет потерю normal KE из импульса и
эффективной обратной массы, распределяет её между отвечающими REV и накапливает
в новом `EnergyLedger::contact_restitution_heat`. Добавлены unit test на
импульсный ответ/тепло и integration test `tests/contact_energy.rs`: два
свободных железных REV сталкиваются через полный `solver::step`; проверяются
импульс, масса и сумма кинетической и внутренней энергий. Счётчик добавлен в
экспериментальные diagnostics `F`, support capture и `contact_probe`, README
документирует API. `VOXEL_SOLVER_BACKEND=cpu cargo test --no-default-features`
прошёл по всем library, binary, integration и doc test targets; `cargo check
--release --no-default-features --all-targets`, `cargo fmt --check` и
`git diff --check` также прошли. Полная энергетика контакта и разрушения всё
ещё не замкнута.

### Completed physical normal-impact capture — 14 October

The capture source has been rerun at 293.15 K after adding f64 sub-ULP thermal
energy accumulation to `Voxel`; the initial 1 K run is retained only as a
diagnostic in `contact-energy-low-internal.csv`. The finalized run uses two iron
REV, opposing ±0.01 m/s X velocities, no gravity, `dt=0.01 s`, 4 substeps,
friction 0, restitution 0.25. It completed 1 760 steps / 17.6 s physical with
177 actual states. Impact occurred at step 100. Ledger values: restitution heat
`7.378123729e-7 J`, friction heat `0`, mass `0.01573999785 kg`, momentum error
`0`, observed internal+kinetic energy error `0 J`. `Voxel::add_internal_energy_j`
stores thermal energy below the f32 state ULP in a serialized f64 remainder.
GIF and MP4 contain 177 frames at 1600×1000 and 10 fps (17.7 s media); MP4 full
decode passed. Live capture endpoint is `:8767`. Rechecking the existing
selector `:8766` showed it had loaded the previous catalog and did not expose
`contact-energy`, despite the JSON being updated. I left that server running
and started selector `:8771`; its experiment list includes `contact-energy`,
and status/GIF/MP4 endpoints all return HTTP 200. This is a free-REV collision
capture, not a fracture-generated fragment collision experiment.

### Agent 1: physical fracture-generated fragment collision

Built and ran `fragments_collision_capture` on the CPU backend. A three-REV iron
chain undergoes cyclic subcritical fatigue loading for a full 17,600 steps
(`dt=1 ms`, 1 solver substep, 17.6 s, 177 saved states); both bonds fail at step
969, and the newly free endpoint collides with its neighbor through ordinary
solver contact at step 970. Settings: gravity off, fatigue cycles 30, exponent
1, threshold 0.1, restitution 0.25, friction 0. Peak stress ratio is 0.3805.
Mass is `0.02360999677 kg`, post-impulse momentum error is
`1.455e-11 kg*m/s`, total kinetic+internal energy error is `0 J`, and final
restitution heat is `7.000165470e-7 J`. COM moved from `x=0.0150 m` to
`x=-0.05359 m`; GIF/MP4 follow the group and show this coordinate. Both files
have 177 frames at 1600×1000, 10 fps, 17.7 s, and the MP4 decodes fully. Direct
viewer :8765 and catalog selector :8774 status/GIF/MP4 return HTTP 200. This is
a small 3-REV mechanics capture, not an FPS benchmark. Later CUDA-requested runs
logged relative PCG residual `5.521e7` at tolerance `1e-5` and explicitly fell
back to CPU PCG; the capture is not evidence of GPU physical simulation.
The momentum check starts after the applied launch impulse and is for the full
three-REV assembly; per-cluster mass/momentum ledgers remain open.

— Агент 1
