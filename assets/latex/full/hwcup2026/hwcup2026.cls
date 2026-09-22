% hwcup2026.cls
% Unofficial LaTeX reproduction of the 2026 "Huawei Cup" 23rd China Postgraduate
% Mathematical Contest in Modeling formatting rules + official Attachment 3 layout.
% Compile with XeLaTeX.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hwcup2026}[2026/09/22 strict reproduction]

\LoadClass[UTF8,zihao=-4,a4paper,fontset=none]{ctexart}
\RequirePackage{iftex}
\ifXeTeX\else
  \ClassError{hwcup2026}{Please compile with XeLaTeX}{The template relies on system Chinese fonts.}
\fi

\RequirePackage{fontspec}
\RequirePackage{xeCJK}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{eso-pic}
\RequirePackage{fancyhdr}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{etoolbox}
\RequirePackage{amsmath,amssymb,mathtools}
\RequirePackage{booktabs,array,longtable,multirow}
\RequirePackage{enumitem}
\RequirePackage{url}
\RequirePackage{hyperref}

% ---- Page geometry copied from the official Attachment 3 Word file ----
% Word twips: top 1702, bottom 1048, left 1276, right 1274, footer 992.
\geometry{
  paper=a4paper,
  top=30.02mm,
  bottom=18.49mm,
  left=22.51mm,
  right=22.47mm,
  headheight=0pt,
  headsep=0pt
}

% ---- Fonts ----
% Exact submission mode: SimSun + SimHei (standard Windows/Office Chinese fonts).
% The fixed decorative cover/abstract-header typography is preserved as images
% extracted from the official Attachment 3, so STXinwei/LiSu are not required.
\newif\ifhwexactfonts
\IfFontExistsTF{SimSun}{%
  \IfFontExistsTF{SimHei}{\hwexactfontstrue}{\hwexactfontsfalse}%
}{\hwexactfontsfalse}

\ifhwexactfonts
  \setCJKmainfont[AutoFakeBold=2.2]{SimSun}
  \setCJKsansfont{SimHei}
  \newCJKfontfamily\hwsong[AutoFakeBold=2.2]{SimSun}
  \newCJKfontfamily\hwhei{SimHei}
\else
  % Preview fallback only. For strict final submission, compile on Windows with
  % SimSun and SimHei available.
  \ClassWarningNoLine{hwcup2026}{SimSun/SimHei not found; using Noto CJK fallback. Final strict submission should use SimSun/SimHei}
  \setCJKmainfont[AutoFakeBold=2.2]{Noto Serif CJK SC}
  \setCJKsansfont{Noto Sans CJK SC}
  \newCJKfontfamily\hwsong[AutoFakeBold=2.2]{Noto Serif CJK SC}
  \newCJKfontfamily\hwhei{Noto Sans CJK SC}
\fi

\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{Liberation Serif}}

% All normal Chinese text: 小四宋体; official line spacing: single.
\AtBeginDocument{%
  \hwsong\zihao{-4}%
  \setstretch{1.0}%
  \setlength{\parindent}{2em}%
  \setlength{\parskip}{0pt}%
}

% ---- No header; centered Arabic page number from abstract page onward ----
\fancypagestyle{hwmain}{%
  \fancyhf{}
  \fancyfoot[C]{\normalfont\fontsize{9pt}{9pt}\selectfont\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\pagestyle{hwmain}

% ---- Headings ----
% Official: paper title 三号黑体; level-1 headings 四号黑体 centered.
% Other Chinese characters remain 小四宋体.
\titleformat{\section}
  {\centering\hwhei\zihao{4}}
  {\thesection}{0.8em}{}
\titlespacing*{\section}{0pt}{1.1\baselineskip}{0.55\baselineskip}

\titleformat{\subsection}
  {\hwsong\bfseries\zihao{-4}}
  {\thesubsection}{0.7em}{}
\titlespacing*{\subsection}{0pt}{0.8\baselineskip}{0.3\baselineskip}

\titleformat{\subsubsection}
  {\hwsong\bfseries\zihao{-4}}
  {\thesubsubsection}{0.7em}{}
\titlespacing*{\subsubsection}{0pt}{0.6\baselineskip}{0.2\baselineskip}

% ---- Figure/table captions ----
\DeclareCaptionFont{hwxiaosi}{\hwsong\zihao{-4}}
\captionsetup{font=hwxiaosi,labelsep=space,justification=centering,singlelinecheck=false}
\AtBeginEnvironment{tabular}{\hwsong\zihao{-4}}
\AtBeginEnvironment{longtable}{\hwsong\zihao{-4}}

% ---- Lists ----
\setlist{nosep,leftmargin=2em}

% ---- PDF metadata / links ----
\hypersetup{hidelinks,unicode=true}

% ---- User metadata ----
\newcommand{\hwSchool}{}
\newcommand{\hwTeamNo}{}
\newcommand{\hwMemberOne}{}
\newcommand{\hwMemberTwo}{}
\newcommand{\hwMemberThree}{}
\newcommand{\hwPaperTitle}{}
\newcommand{\hwKeywords}{}

\newcommand{\school}[1]{\renewcommand{\hwSchool}{#1}}
\newcommand{\teamno}[1]{\renewcommand{\hwTeamNo}{#1}}
\newcommand{\memberone}[1]{\renewcommand{\hwMemberOne}{#1}}
\newcommand{\membertwo}[1]{\renewcommand{\hwMemberTwo}{#1}}
\newcommand{\memberthree}[1]{\renewcommand{\hwMemberThree}{#1}}
\newcommand{\papertitle}[1]{\renewcommand{\hwPaperTitle}{#1}}
\newcommand{\keywords}[1]{\renewcommand{\hwKeywords}{#1}}

% ---- Official cover ----
% The background is a 300-dpi rendering of page 1 of the official Attachment 3
% supplied by the user. This preserves all four logos, fixed typography, rules,
% and the template's visible page number 0 exactly.
\newcommand{\makehwcover}{%
  \clearpage
  \thispagestyle{empty}%
  \AddToShipoutPictureBG*{%
    \AtPageLowerLeft{\includegraphics[width=\paperwidth,height=\paperheight]{assets/official-cover.png}}%
  }%
  \begin{tikzpicture}[remember picture,overlay]
    % Empty cells in the Word template inherit 18 pt; use normal Song font.
    \node[anchor=south,align=center,text width=11.5cm] at ([xshift=12.10cm,yshift=-14.78cm]current page.north west)
      {\hwsong\fontsize{18pt}{20pt}\selectfont \hwSchool};
    \node[anchor=south,align=center,text width=11.5cm] at ([xshift=12.10cm,yshift=-16.83cm]current page.north west)
      {\hwsong\fontsize{18pt}{20pt}\selectfont \hwTeamNo};
    \node[anchor=south,align=center,text width=9.6cm] at ([xshift=12.65cm,yshift=-18.00cm]current page.north west)
      {\hwsong\fontsize{18pt}{20pt}\selectfont \hwMemberOne};
    \node[anchor=south,align=center,text width=9.6cm] at ([xshift=12.65cm,yshift=-19.16cm]current page.north west)
      {\hwsong\fontsize{18pt}{20pt}\selectfont \hwMemberTwo};
    \node[anchor=south,align=center,text width=9.6cm] at ([xshift=12.65cm,yshift=-20.31cm]current page.north west)
      {\hwsong\fontsize{18pt}{20pt}\selectfont \hwMemberThree};
  \end{tikzpicture}%
  \null\clearpage
  \setcounter{page}{1}%
  \pagestyle{hwmain}%
}

% ---- Abstract page ----
% Header and fixed labels come from the official Attachment 3 render, preserving
% its decorative typography. User content follows the 2026 official rules.
\newcommand{\hwabstractbegin}{%
  \clearpage
  \thispagestyle{hwmain}%
  \AddToShipoutPictureFG*{%
    \AtPageLowerLeft{\raisebox{20.98cm}[0pt][0pt]{\includegraphics[width=\paperwidth]{assets/abstract-header.png}}}%
  }%
  % Cursor begins at the 30.02 mm top margin. Move to the title row (~9.9 cm from page top).
  \vspace*{6.56cm}%
  \noindent
  \raisebox{-0.10cm}{\includegraphics[height=0.83cm]{assets/label-title.png}}%
  \hspace{0.16cm}%
  \begin{minipage}[b]{10.35cm}
    \centering\hwhei\zihao{3}\hwPaperTitle\par
    \vspace{1pt}\hrule height 0.45pt
  \end{minipage}
  \par\vspace{1.55cm}
  \begin{center}
    \includegraphics[height=0.90cm]{assets/label-abstract.png}
  \end{center}
  \vspace{0.42cm}
  \hwsong\zihao{-4}\setstretch{1.0}\setlength{\parindent}{2em}%
}

\newcommand{\hwabstractend}{%
  \par\vspace{0.65\baselineskip}%
  \noindent\raisebox{-0.11cm}{\includegraphics[height=0.80cm]{assets/label-keywords.png}}%
  \hspace{0.12cm}{\hwsong\zihao{-4}\hwKeywords}%
  \par
  \clearpage
}

\newenvironment{hwabstract}{\hwabstractbegin}{\hwabstractend}

% ---- Convenience environments ----
\newenvironment{notation}{%
  \begin{center}\begin{tabular}{>{\centering\arraybackslash}p{0.18\textwidth}p{0.72\textwidth}}
  \toprule
  符号 & 含义\\
  \midrule
}{%
  \bottomrule
  \end{tabular}\end{center}
}

% Reference list should be in citation order and use the contest's specified
% textual forms; users can write entries manually in thebibliography.
\renewcommand{\refname}{参考文献}
